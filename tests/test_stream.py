import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_nuts import (
    AdaptiveWindows,
    DiagonalAdaptation,
    FixedWindows,
    Jobs,
    StructuredAdaptation,
    sample,
    sample_batches,
    warmup,
)


def density(q):
    return -jnp.dot(q, q) / 2


@pytest.mark.parametrize("adapt", [False, True])
def test_earliest_and_every_available_draw_at_budget(adapt):
    n = 3
    jobs = Jobs(
        jax.random.split(jax.random.key(43), n),
        np.zeros((n, 2)),
        np.array([0.003, 0.2, 1.0]),
        np.ones((n, 2)),
    )
    options = {}
    delay = np.zeros(n, dtype=int)
    frozen = jobs
    if adapt:
        options["schedule"] = FixedWindows((8,), 4, clock="transitions")
        trained = warmup(density, jobs, max_depth=5, **options)
        frozen = trained.jobs
        delay = trained.num_evaluations.sum(1)
    reference = sample(density, frozen, 350, max_depth=5)
    ends = delay[:, None] + reference.num_integration_steps.cumsum(axis=1)
    batches = list(
        sample_batches(
            density,
            jobs,
            max_depth=5,
            buffer_size=12,
            max_rounds=317,
            rounds_per_batch=53,
            **options,
        )
    )
    first = next(b for b in batches if len(b.records.job))
    assert first.num_rounds == ends[:, 0].min()
    if adapt:
        assert first.num_rounds < delay.max()
    assert batches[-1].num_rounds == 317
    seen = np.zeros(n, dtype=int)
    for b in batches:
        r = b.records
        assert len(r.job) <= 12
        np.testing.assert_allclose(
            r.position, reference.position[r.job, r.draw], atol=1e-11
        )
        np.testing.assert_array_equal(
            r.num_integration_steps, reference.num_integration_steps[r.job, r.draw]
        )
        np.testing.assert_array_equal(b.completion_rounds, ends[r.job, r.draw])
        seen += np.bincount(r.job, minlength=n)
        np.testing.assert_array_equal(seen, (ends <= b.num_rounds).sum(1))
    assert seen.max() > seen.min()


def test_adaptive_endpoint_and_partial_checkpoint():
    jobs = Jobs(
        jax.random.split(jax.random.key(12), 2),
        np.zeros((2, 2)),
        np.array([0.2, 0.5]),
        np.ones((2, 2)),
    )
    schedule = AdaptiveWindows(
        first_window=16, tolerance=100, max_windows=5, final_window=8
    )
    solved = warmup(density, jobs, schedule=schedule, max_depth=4)
    delays = solved.num_evaluations.sum(1)
    expected = sample(density, solved.jobs, 100, max_depth=4)
    stop = int(delays.min() + 60)
    batches = list(
        sample_batches(
            density,
            jobs,
            schedule=schedule,
            max_depth=4,
            max_rounds=stop,
            rounds_per_batch=13,
            buffer_size=7,
        )
    )
    checks = delays[:, None] + expected.num_integration_steps.cumsum(1)
    seen = np.zeros(2, int)
    for batch in batches:
        rec = batch.records
        seen += np.bincount(rec.job, minlength=2)
        np.testing.assert_array_equal(batch.completion_rounds, checks[rec.job, rec.draw])
        np.testing.assert_allclose(
            rec.position, expected.position[rec.job, rec.draw], atol=1e-11
        )
        np.testing.assert_array_equal(seen, (checks <= batch.num_rounds).sum(1))
    assert batches[-1].num_rounds == stop


def test_unstable_job_is_reported_once_while_healthy_job_keeps_sampling():
    def scaled_density(q, precision):
        return precision * density(q)

    jobs = Jobs(
        jax.random.split(jax.random.key(12), 2),
        np.zeros((2, 2)),
        np.full(2, 0.2),
        np.ones((2, 2)),
        dataset=np.arange(2),
    )
    # A flat score has zero variance, so job 1 can never establish a metric.
    schedule = AdaptiveWindows(
        first_window=10,
        growth=1,
        clock="transitions",
        max_windows=3,
        tolerance=100,
        final_window=4,
    )
    options = dict(schedule=schedule, max_depth=3, buffer_size=5)
    events = []
    mixed = list(
        sample_batches(
            scaled_density,
            jobs,
            data=jnp.array([1.0, 0.0]),
            max_rounds=400,
            on_warmup=events.extend,
            **options,
        )
    )
    np.testing.assert_array_equal(np.concatenate([b.failed_jobs for b in mixed]), [1])
    report = next(b for b in mixed if len(b.failed_jobs))
    (failure,) = [e for e in events if e.failed]
    assert failure.window_closed and failure.job == 1 and failure.round == report.num_rounds
    assert failure.stable is False and not np.isfinite(failure.stability_range)
    assert all(e.round <= failure.round for e in events if e.job == 1)
    assert report.num_rounds < 400
    assert any(len(b.records.job) for b in mixed if b.num_rounds > report.num_rounds)
    one = Jobs(*(field[:1] for field in jobs))
    healthy = list(
        sample_batches(scaled_density, one, data=jnp.ones(1), max_rounds=400, **options)
    )
    for batches in (mixed, healthy):
        assert batches[-1].num_rounds == 400
        assert not np.concatenate([b.records.job for b in batches]).any()
    np.testing.assert_array_equal(
        np.concatenate([b.completion_rounds for b in mixed]),
        np.concatenate([b.completion_rounds for b in healthy]),
    )
    np.testing.assert_allclose(
        np.concatenate([b.records.position for b in mixed]),
        np.concatenate([b.records.position for b in healthy]),
        atol=1e-11,
    )

    exhausted = list(sample_batches(scaled_density, jobs, data=jnp.zeros(2), **options))
    assert sorted(np.concatenate([b.failed_jobs for b in exhausted])) == [0, 1]
    assert all(len(b.records.job) == 0 for b in exhausted)


def correlated(q):
    return -0.5 * (q[0] ** 2 + (q[1] + 0.75 * q[0]) ** 2)


def test_callback_reports_the_stopping_monitor_at_each_actual_boundary():
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
    options = dict(
        schedule=schedule, diagonal=DiagonalAdaptation(evidence="nuts"), max_depth=4
    )
    trained = warmup(correlated, jobs, **options)
    options.update(buffer_size=6, max_rounds=180, rounds_per_batch=31)
    baseline = list(sample_batches(correlated, jobs, **options))
    assert next(b.num_rounds for b in baseline if len(b.records.job)) < max(
        trained.num_evaluations.sum(1)
    )

    for detail in (False, True):
        events = []
        stream = sample_batches(
            correlated,
            jobs,
            on_warmup=events.extend,
            include_window_evidence=detail,
            **options,
        )
        for batch, expected in zip(stream, baseline, strict=True):
            assert all(e.round <= batch.num_rounds for e in events)
            for actual, control in zip(batch.records, expected.records, strict=True):
                np.testing.assert_array_equal(actual, control)
            np.testing.assert_array_equal(
                batch.completion_rounds, expected.completion_rounds
            )
            np.testing.assert_array_equal(batch.failed_jobs, expected.failed_jobs)
            assert batch.num_rounds == expected.num_rounds

        assert [(e.round, e.job) for e in events] == sorted(
            {(e.round, e.job) for e in events}
        )
        for job in range(3):
            closed = [e for e in events if e.job == job and e.window_closed]
            assert closed[-1].frozen and not any(e.frozen for e in closed[:-1])
            assert not any(e.failed for e in closed)
            counts = trained.num_evaluations[job]
            assert [e.window for e in closed] == list(np.flatnonzero(counts))
            previous = None
            for event in closed:
                w = event.window
                assert event.round == counts[: w + 1].sum()
                assert event.num_evaluations == counts[w]
                assert event.num_transitions == trained.num_transitions[job, w]
                divergences = sum(
                    e.divergent for e in events if e.job == job and e.window == w
                )
                assert (
                    event.num_divergences == divergences == trained.num_divergences[job, w]
                )
                assert event.evidence_requested == detail
                if event.frozen:
                    assert (
                        event.evidence_source
                        is event.stability_range
                        is event.stable
                        is None
                    )
                    assert event.evidence is None
                    continue
                assert event.evidence_source == "selected"
                if w == 0:
                    assert event.stable is event.stability_range is None
                else:
                    assert event.stable == (
                        event.stability_range < np.log(schedule.tolerance)
                    )
                if not detail:
                    assert event.evidence is None
                    continue
                ev = event.evidence
                assert ev.weight == event.num_transitions
                precision = np.array([[1.5625, 0.75], [0.75, 1.0]])
                np.testing.assert_allclose(ev.gradient_mean, -precision @ ev.position_mean)
                shape = (np.log(ev.position_variance) - np.log(ev.gradient_variance)) / 2
                if previous is not None:
                    np.testing.assert_allclose(
                        event.stability_range, np.ptp((shape - previous) / 2)
                    )
                previous = shape


def test_simultaneous_divergences_before_a_window_closes():
    n = 67
    jobs = Jobs(
        jax.random.split(jax.random.key(42), n),
        np.ones((n, 2)),
        np.full(n, 1e4),
        np.ones((n, 2)),
    )
    kwargs = dict(schedule=FixedWindows((2,), 2, clock="transitions"), max_depth=1)
    groups = []
    batches = list(
        sample_batches(density, jobs, max_rounds=1, on_warmup=groups.append, **kwargs)
    )
    assert [len(g) for g in groups] == [64, 3]
    assert len(batches) == 1 and len(batches[0].records.job) == 0
    events = [event for group in groups for event in group]
    assert [e.job for e in events] == list(range(n))
    for e in events:
        assert e.round == e.num_divergences == e.num_evaluations == e.num_transitions == 1
        assert e.divergent and not (e.window_closed or e.failed or e.frozen)
        assert e.stability_range is e.stable is e.evidence is None

    def stop(events):
        raise RuntimeError("observer stopped")

    with pytest.raises(RuntimeError, match="observer stopped"):
        next(sample_batches(density, jobs, on_warmup=stop, **kwargs))
    assert list(sample_batches(density, jobs, on_warmup=stop, max_rounds=0, **kwargs)) == []
    assert list(sample_batches(density, jobs, on_warmup=stop, max_rounds=1, max_depth=1))
    with pytest.raises(ValueError, match="on_warmup"):
        next(sample_batches(density, jobs, include_window_evidence=True, **kwargs))


@pytest.mark.parametrize("structured", [False, True])
def test_summary_and_evidence_availability_for_other_warmup_policies(structured):
    jobs = Jobs(
        jax.random.split(jax.random.key(7), 2),
        np.zeros((2, 2)),
        np.full(2, 0.2),
        np.ones((2, 2)),
    )
    schedule = (
        AdaptiveWindows(
            first_window=10,
            growth=1,
            clock="transitions",
            tolerance=100,
            max_windows=3,
            final_window=3,
        )
        if structured
        else FixedWindows((10,), 3, clock="transitions")
    )
    events = []
    list(
        sample_batches(
            correlated,
            jobs,
            schedule=schedule,
            structured=StructuredAdaptation(capacity=8, max_reflections=0)
            if structured
            else None,
            max_depth=2,
            max_rounds=140,
            on_warmup=events.extend,
            include_window_evidence=True,
        )
    )
    for job in range(2):
        closed = [e for e in events if e.job == job and e.window_closed]
        assert closed[0].stable is None and closed[-1].frozen
        if structured:
            assert closed[-2].stable is True
        else:
            assert [e.window for e in closed] == [0, 1]
            np.testing.assert_array_equal(
                [e.round for e in closed], np.cumsum([e.num_evaluations for e in closed])
            )
            assert all(e.stable is None for e in closed)
    for e in events:
        assert e.evidence_requested
        assert e.evidence_source is e.stability_range is e.evidence is None
