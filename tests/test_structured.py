import jax
import jax.numpy as jnp
import numpy as np
import pytest

from batched_nuts import (
    AdaptiveWindows,
    FixedWindows,
    Jobs,
    MetricFactor,
    StructuredAdaptation,
    sample,
    sample_batches,
    warmup,
)
from batched_nuts.metric import _apply, _momentum, _velocity
from batched_nuts.structured import _prepare_factor, _select, _window


def dense(factor):
    outer, rows, lower, scale = map(np.asarray, factor)
    d = len(scale)
    out = np.diag(outer)
    for v in rows:
        out = out @ (np.eye(d) - 2 * np.outer(v, v))
    block = np.eye(d)
    r = len(lower)
    block[:r, :r] = lower
    return out @ block @ np.diag(scale)


def row(factor, i):
    return jax.tree.map(lambda x: jnp.asarray(x[i]), factor)


def test_compact_factor_and_all_four_operations():
    rng = np.random.default_rng(10)
    diag = np.array([[1.0, 2.0, 4.0, 3.0], [3.0, 1.0, 1.5, 2.0]])
    basis = rng.normal(size=(2, 4, 2))
    core = np.array([[[1.0, -0.1], [-0.1, -0.02]], [[0.1, 0.07], [0.07, 0.2]]])
    factors = MetricFactor.from_low_rank(diag, basis, core)
    x = rng.normal(size=(5, 4))
    for i in range(2):
        f = row(factors, i)
        L = dense(f)
        np.testing.assert_allclose(
            L @ L.T, np.diag(diag[i]) + basis[i] @ core[i] @ basis[i].T
        )
        scale = np.array([1.1, 0.4, 3.5, 2.6])
        f = f._replace(scale=scale)
        L = L @ np.diag(scale)
        for transpose in (False, True):
            matrix = L.T if transpose else L
            np.testing.assert_allclose(_apply(f, x, transpose=transpose), x @ matrix.T)
            np.testing.assert_allclose(
                _apply(f, x, transpose=transpose, solve=True),
                np.linalg.solve(matrix, x.T).T,
            )
        np.testing.assert_allclose(_velocity(f, x), x @ L @ L.T)
        np.testing.assert_allclose(_momentum(f, x[0]), np.linalg.solve(L.T, x[0]))


def test_same_window_reverses_score_and_unmeasurable_window_keeps_metric():
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    L = np.linalg.cholesky(cov)
    z = np.sqrt(0.9) * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    x = z @ L.T
    g = -np.linalg.solve(cov, x.T).T
    initial = MetricFactor(jnp.ones(2), jnp.empty((0, 2)), jnp.array(L), jnp.ones(2))
    window = _window(initial, 4)._replace(
        points=jnp.asarray(np.column_stack((x, g))), count=4
    )
    for score, expected in [("mean_product", cov), ("squared_log", 0.6 * np.eye(2))]:
        metric, _ = _select(window, initial, StructuredAdaptation(score, 4, 0), 2.0)
        actual = dense(metric)
        np.testing.assert_allclose(actual @ actual.T, expected, atol=1e-12)
    metric, stable = _select(
        window._replace(count=0), initial, StructuredAdaptation(max_reflections=0), 2.0
    )
    np.testing.assert_array_equal(dense(metric), L)
    assert not stable


def test_reflection_recovers_orientation_with_unchanged_marginals():
    rho = 0.995
    cov = np.array([[1.0, rho], [rho, 1.0]])
    x = np.sqrt(1.5) * np.vstack((np.eye(2), -np.eye(2))) @ np.linalg.cholesky(cov).T
    g = -np.linalg.solve(cov, x.T).T
    initial = row(_prepare_factor(np.ones((1, 2)), 1), 0)
    window = _window(initial, 4)._replace(
        points=jnp.asarray(np.column_stack((x, g))), count=4
    )
    fit, stable = _select(
        window, initial, StructuredAdaptation(capacity=4, max_reflections=1), 2.0
    )
    np.testing.assert_allclose(dense(fit) @ dense(fit).T, cov, atol=1e-12)
    assert not stable
    _, stable = _select(
        window, fit, StructuredAdaptation(capacity=4, max_reflections=1), 2.0
    )
    assert stable


def normal(x):
    return -0.5 * x @ x


def test_factor_trajectories_match_explicit_change_of_coordinates_and_refill():
    # A dense oracle target in whitened coordinates shares the random keys with
    # the original target. Agreement checks momentum, leapfrog and both U-turns.
    basis = np.array([[[1.0], [0.8]], [[0.5], [-1.0]], [[0.8], [0.8]]])
    f = MetricFactor.from_low_rank(np.ones((3, 2)), basis, np.ones((3, 1, 1)) * 3.0)
    matrices = np.stack([dense(row(f, i)) for i in range(3)])
    z = np.array([[0.5, 0.9], [-0.4, 0.6], [0.2, 0.1]])
    jobs = Jobs(
        jax.random.split(jax.random.key(17), 3),
        np.einsum("ijk,ik->ij", matrices, z),
        np.array([0.25, 0.2, 0.35]),
        f,
    )
    actual = sample(normal, jobs, 25, num_lanes=2, max_depth=5)

    def whitened(x, L):
        return normal(L @ x)

    ref = sample(
        whitened,
        jobs._replace(position=z, inverse_mass=np.ones_like(z), dataset=np.arange(3)),
        25,
        data=jnp.asarray(matrices),
        num_lanes=2,
        max_depth=5,
    )
    np.testing.assert_allclose(
        actual.position, np.einsum("ijk,itk->itj", matrices, ref.position), atol=1e-11
    )
    np.testing.assert_array_equal(actual.num_integration_steps, ref.num_integration_steps)
    np.testing.assert_array_equal(actual.is_turning, ref.is_turning)


def test_stream_uses_same_per_chain_frozen_factors_and_keys_after_refill():
    init = MetricFactor.from_low_rank(
        np.ones((3, 2)),
        np.array([[[1.0], [1.0]], [[2.0], [-1.0]], [[0.0], [1.0]]]),
        np.ones((3, 1, 1)),
    )
    jobs = Jobs(
        jax.random.split(jax.random.key(47), 3), np.ones((3, 2)), np.full(3, 0.24), init
    )
    kwargs = dict(
        structured=StructuredAdaptation(capacity=7, max_reflections=1),
        schedule=FixedWindows((16, 17), 4, clock="transitions"),
        max_depth=3,
    )
    fit = warmup(normal, jobs, num_lanes=2, **kwargs)
    draw = sample(normal, fit.jobs, 180, max_depth=3)
    batches = list(
        sample_batches(normal, jobs, rounds_per_batch=23, max_rounds=150, **kwargs)
    )
    ends = fit.num_evaluations.sum(axis=1)[:, None] + draw.num_integration_steps.cumsum(
        axis=1
    )
    assert sum(len(b.records.job) for b in batches) == np.sum(ends <= 150) > 0
    for b in batches:
        idx = b.records.job, b.records.draw
        np.testing.assert_array_equal(b.completion_rounds, ends[idx])
        np.testing.assert_allclose(b.records.position, draw.position[idx], atol=1e-11)


def test_adaptive_stream_keeps_failure_local():
    def target(x, stuck):
        return jnp.where(stuck, 0.0, normal(x))

    jobs = Jobs(
        jax.random.split(jax.random.key(5), 2),
        np.ones((2, 2)),
        np.full(2, 0.2),
        np.ones((2, 2)),
        np.arange(2),
    )
    batches = list(
        sample_batches(
            target,
            jobs,
            data=jnp.array([True, False]),
            structured=StructuredAdaptation(capacity=8),
            schedule=AdaptiveWindows(16, 2, 2.0, 3, 4, clock="transitions"),
            max_depth=2,
            max_rounds=350,
        )
    )
    failed = np.concatenate([b.failed_jobs for b in batches])
    drawn = np.concatenate([b.records.job for b in batches])
    np.testing.assert_array_equal(failed, [0])
    assert set(drawn) == {1}


def test_reject_invalid_factor_before_sampling():
    jobs = Jobs(
        jax.random.split(jax.random.key(0), 1),
        np.ones((1, 2)),
        np.ones(1),
        MetricFactor(
            np.ones((1, 2)), np.ones((1, 1, 2)), np.empty((1, 0, 0)), np.ones((1, 2))
        ),
    )
    with pytest.raises(ValueError, match="unit vectors"):
        sample(normal, jobs, 3)
    with pytest.raises(ValueError, match="score"):
        StructuredAdaptation(score="other")


def test_only_closing_lanes_install_a_fit_in_a_partial_batch():
    from batched_nuts.structured import _fit_due

    width = 19
    policy = StructuredAdaptation(capacity=4, max_reflections=0)
    factor = jax.tree.map(jnp.asarray, _prepare_factor(np.ones((width, 2)), 0))
    windows = jax.vmap(lambda f: _window(f, 4))(factor)
    x = jnp.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0], [0.0, -2.0]])
    points = jnp.column_stack((x, -x * jnp.array([4.0, 0.25])))
    windows = windows._replace(
        points=jnp.broadcast_to(points, windows.points.shape), count=jnp.full(width, 4)
    )
    due = jnp.arange(width) % 2 == 0
    fitted, stable = jax.jit(
        jax.vmap(lambda w, m, d: _fit_due(w, m, d, policy, 2.0), axis_name="lanes")
    )(windows, factor, due)
    np.testing.assert_allclose(
        fitted.scale, np.where(np.asarray(due)[:, None], [0.5, 2.0], [1.0, 1.0])
    )
    assert not np.any(stable)
