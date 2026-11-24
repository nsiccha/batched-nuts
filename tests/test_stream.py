import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_nuts import AdaptiveWindows, FixedWindows, Jobs, sample, sample_batches, warmup


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
    mixed = list(
        sample_batches(
            scaled_density, jobs, data=jnp.array([1.0, 0.0]), max_rounds=400, **options
        )
    )
    np.testing.assert_array_equal(np.concatenate([b.failed_jobs for b in mixed]), [1])
    report = next(b for b in mixed if len(b.failed_jobs))
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
