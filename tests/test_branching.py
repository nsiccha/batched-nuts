"""Children match separate chains started from the same parent endpoint and tuning."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import batched_nuts.stream as stream
from batched_nuts import (
    AdaptiveWindows,
    Branch,
    Branching,
    FixedWindows,
    Jobs,
    MetricFactor,
    StructuredAdaptation,
    sample,
    sample_batches,
    warmup,
)


def density(q):
    return -0.5 * (q[0] ** 2 + (q[1] + 0.75 * q[0]) ** 2)


def draws(batches, job):
    records = jax.tree.map(lambda *x: np.concatenate(x), *(b.records for b in batches))
    rows = records.job == job
    rounds = np.concatenate([b.completion_rounds for b in batches])[rows]
    return jax.tree.map(lambda x: x[rows], records), rounds


def compare_child(density, trained, start, origin, key, batches, depth, data=None):
    p = origin.parent
    jobs = Jobs(
        jax.random.fold_in(key, origin.job)[None],
        start[None],
        trained.step_size[p : p + 1],
        jax.tree.map(lambda x: x[p : p + 1], trained.inverse_mass),
        None if data is None else trained.dataset[p : p + 1],
    )
    ref = sample(density, jobs, 150, max_depth=depth, data=data)
    rec, rounds = draws(batches, origin.job)
    ends = origin.admission_round + np.cumsum(ref.num_integration_steps[0])
    select = (ends <= batches[-1].num_rounds) & (np.arange(len(ends)) >= origin.settle)
    assert len(rec.job) > 0
    np.testing.assert_array_equal(rounds, ends[select])
    np.testing.assert_array_equal(rec.draw, np.arange(select.sum()))
    for name in ref._fields[:6]:
        np.testing.assert_allclose(
            getattr(rec, name), getattr(ref, name)[0, select], atol=1e-10, rtol=1e-10
        )


def test_branch_from_freeze_and_from_draw_while_incumbents_continue(monkeypatch):
    jobs = Jobs(
        jax.random.split(jax.random.key(12), 3),
        np.zeros((3, 2)),
        np.array([0.003, 0.2, 1.0]),
        np.ones((3, 2)),
    )
    schedule = AdaptiveWindows(
        first_window=8,
        growth=1,
        clock="transitions",
        tolerance=100,
        max_windows=4,
        final_window=4,
        evidence="selected",
    )
    kwargs = dict(schedule=schedule, max_depth=4)
    trained = warmup(density, jobs, **kwargs)
    freezes = trained.num_evaluations.sum(1)
    kwargs.update(buffer_size=10, max_rounds=240, rounds_per_batch=17)
    reference = list(sample_batches(density, jobs, **kwargs))
    widths = []
    original_mixed = stream._mixed_loop

    def traced_mixed(*args, **kwargs):
        kernel = original_mixed(*args, **kwargs)

        def run(state, *args):
            widths.append(state.lanes.active.size)
            return kernel(state, *args)

        return run

    monkeypatch.setattr(stream, "_mixed_loop", traced_mixed)

    def choose(batch, parents, room):
        p = parents[0]
        if room == 2:
            assert batch.num_rounds == freezes.min()
            assert p.warmup_rounds[-1] == freezes[p.job]
            assert len(p.warmup_rounds) == 2
            return [Branch(p.job, p.warmup_rounds[-1], settle=3)]
        if p.draw_round is not None and batch.num_rounds >= 145:
            return [Branch(p.job, p.draw_round)]

    branch = Branching(jax.random.key(721), 1, 5, choose, warmup_history=2)
    events = []
    batches = list(
        sample_batches(density, jobs, branching=branch, on_warmup=events.extend, **kwargs)
    )
    assert list(dict.fromkeys(widths)) == [3, 4]
    origins = [o for b in batches for o in b.branches]
    assert [o.job for o in origins] == [3, 4]
    assert origins[0].admission_round < max(freezes) < origins[1].admission_round
    assert [o.warmup for o in origins] == [True, False]
    assert sorted((e.job, e.round) for e in events if e.frozen) == list(enumerate(freezes))
    for job in range(3):
        (rec, r), (control, cr) = draws(batches, job), draws(reference, job)
        np.testing.assert_array_equal(r, cr)
        for actual, expected in zip(rec, control, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    for origin in origins:
        rec, r = draws(reference, origin.parent)
        start = (
            trained.jobs.position[origin.parent]
            if origin.warmup
            else rec.position[r == origin.source_round][0]
        )
        compare_child(density, trained.jobs, start, origin, branch.key, batches, 4)


def target(q, center):
    return -0.5 * jnp.dot(q - center, q - center)


def test_earlier_warmup_endpoint_with_parent_dataset_and_factor():
    keys = jax.random.split(jax.random.key(14), 3)
    datasets = np.array([1, 0, 1])
    data = jax.device_put(np.array([[0.0, 0.0], [4.0, -3.0]]), jax.devices()[-1])
    metric = MetricFactor.from_low_rank(
        np.ones((3, 2)), np.ones((3, 2, 1)) * 0.2, np.ones((3, 1, 1))
    )
    jobs = Jobs(keys, np.zeros((3, 2)), np.full(3, 0.25), metric, datasets)
    config = dict(
        data=data,
        schedule=FixedWindows((1,), 3, clock="transitions"),
        structured=StructuredAdaptation(max_reflections=0, capacity=4),
        max_depth=3,
    )
    trained = warmup(target, jobs, **config)
    # End a separate warmup after its second trajectory. This selects exactly
    # the position visited inside the full run's three-transition final window.
    shorter = dict(config, schedule=FixedWindows((1,), 1, clock="transitions"))
    interior = warmup(target, jobs, **shorter)

    def choose(batch, parents, room):
        assert [(p.job, p.dataset) for p in parents] == [(0, 1), (2, 1)]
        p = parents[-1]
        assert len(p.warmup_rounds) == 4
        assert p.warmup_rounds[1] == interior.num_evaluations[p.job].sum()
        assert p.warmup_rounds[-1] == trained.num_evaluations[p.job].sum()
        return [Branch(p.job, p.warmup_rounds[1], settle=2)] * room

    branch_key = jax.device_put(jax.random.key(7), jax.devices()[0])
    branch = Branching(branch_key, 2, 5, choose, warmup_history=4)
    batches = list(sample_batches(target, jobs, branching=branch, max_rounds=120, **config))
    origins = [o for b in batches for o in b.branches]
    assert len(origins) == 2
    for origin in origins:
        assert origin.dataset == 1 and origin.parent == 2
        assert origin.source_round < origin.admission_round
        compare_child(
            target,
            trained.jobs,
            interior.jobs.position[2],
            origin,
            branch.key,
            batches,
            3,
            data,
        )
    assert not np.array_equal(draws(batches, 3)[0].position, draws(batches, 4)[0].position)


def test_no_cross_target_gate_and_no_admission_at_deadline():
    jobs = Jobs(
        jax.random.split(jax.random.key(18), 2),
        np.zeros((2, 2)),
        np.ones(2),
        np.ones((2, 2)),
        dataset=np.arange(2),
    )

    def unexpected(*args):
        pytest.fail("no target should be eligible for admission")

    branching = Branching(jax.random.key(81), 2, 3, unexpected)
    args = dict(data=np.zeros((2, 2)), branching=branching, max_depth=1, max_rounds=7)
    batches = list(sample_batches(target, jobs, **args))
    assert all(not b.branches for b in batches)
    assert sum(len(b.records.job) for b in batches) == 14
    args.update(branching=replace(branching, ready_target=1), max_rounds=1)
    assert not next(sample_batches(target, jobs, **args)).branches


def test_partial_and_total_warmup_failure():
    def scaled(q, precision):
        return -0.5 * precision * jnp.dot(q, q)

    jobs = Jobs(
        jax.random.split(jax.random.key(12), 2),
        np.zeros((2, 2)),
        np.full(2, 0.2),
        np.ones((2, 2)),
        np.arange(2),
    )
    schedule = AdaptiveWindows(
        first_window=10,
        growth=1,
        tolerance=100,
        max_windows=3,
        final_window=4,
        clock="transitions",
    )

    def choose(batch, parents, room):
        assert [p.job for p in parents] == [0]
        if parents[0].draw_round is not None:
            return [Branch(0, parents[0].draw_round)]

    args = dict(
        schedule=schedule,
        branching=Branching(jax.random.key(34), 1, 3, choose),
        max_rounds=400,
        max_depth=3,
        rounds_per_batch=51,
    )
    mixed = list(sample_batches(scaled, jobs, data=jnp.array([1.0, 0.0]), **args))
    np.testing.assert_array_equal(np.concatenate([b.failed_jobs for b in mixed]), [1])
    assert len(draws(mixed, 2)[1]) > 0
    exhausted = list(sample_batches(scaled, jobs, data=jnp.zeros(2), **args))
    assert sorted(np.concatenate([b.failed_jobs for b in exhausted])) == [0, 1]
    assert all(not b.branches for b in exhausted)


def test_default_buffer_grows_only_at_admission():
    jobs = Jobs(
        jax.random.split(jax.random.key(33), 1),
        np.zeros((1, 2)),
        np.ones(1),
        np.ones((1, 2)),
    )

    def choose(batch, parents, room):
        return [Branch(0, parents[0].draw_round)] * room

    # Raw two-word keys are accepted alongside typed keys.
    policy = Branching(jax.random.PRNGKey(46), 1, 257, choose)
    first, second = sample_batches(
        density, jobs, branching=policy, max_depth=1, max_rounds=2
    )
    assert first.records.job.tolist() == [0] and len(first.branches) == 256
    np.testing.assert_array_equal(second.records.job, np.arange(257))
    np.testing.assert_array_equal(second.completion_rounds, np.full(257, 2))


def test_invalid_requests_and_finite_limits():
    jobs = Jobs(
        jax.random.split(jax.random.key(19), 1),
        np.zeros((1, 2)),
        np.ones(1),
        np.ones((1, 2)),
    )
    policy = Branching(jax.random.key(2), 1, 2, lambda *args: (), warmup_history=2)
    args = dict(schedule=FixedWindows((1,), 1, clock="transitions"), max_depth=1)
    for requests, message in [
        ([Branch(1, 1)], "eligible frozen"),
        ([Branch(0, 100)], "not available"),
        ([Branch(0, 1)], "settling period"),
        ([Branch(0, 1, -1)], "nonnegative"),
        ([Branch(0, 1, 0)] * 2, "exceed max_chains"),
    ]:

        def choose(batch, parents, room, requests=requests):
            return requests

        with pytest.raises(ValueError, match=message):
            next(
                sample_batches(
                    density, jobs, branching=replace(policy, choose=choose), **args
                )
            )
    with pytest.raises(ValueError, match="buffer_size"):
        next(sample_batches(density, jobs, branching=policy, buffer_size=1, **args))
    with pytest.raises(ValueError, match="ready_target"):
        next(
            sample_batches(density, jobs, branching=replace(policy, ready_target=2), **args)
        )
    assert list(sample_batches(density, jobs, branching=policy, max_rounds=0, **args)) == []
