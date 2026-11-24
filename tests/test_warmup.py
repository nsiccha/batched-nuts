"""Diagonal warmup consumes complete trajectories and hands a fresh key to sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blackjax.adaptation.step_size import dual_averaging_adaptation

from batched_nuts import Jobs, sample, warmup
from batched_nuts.nuts import _advance, _assign, _begin_next, _empty_lanes
from batched_nuts.warmup import _dual, _start_dual

from targets import gaussian_jobs


def test_refill_budgets_metric_and_sampling_keys():
    density, jobs, data = gaussian_jobs(dim=5)
    jobs = jobs._replace(inverse_mass=jnp.ones_like(jobs.position))
    windows = (64, 128, 64)
    full = warmup(density, jobs, data=data, max_depth=5, window_evaluations=windows)
    queued = warmup(
        density, jobs, data=data, max_depth=5, window_evaluations=windows, num_lanes=3
    )
    for field in full._fields[1:]:
        np.testing.assert_array_equal(getattr(queued, field), getattr(full, field))
    for name in Jobs._fields:
        np.testing.assert_allclose(getattr(queued.jobs, name), getattr(full.jobs, name))
    expected = jax.vmap(jax.random.split)(jobs.key)[:, 1]
    np.testing.assert_array_equal(full.jobs.key, jax.random.key_data(expected))
    assert np.all(full.num_evaluations >= windows)
    assert np.all(full.num_evaluations < np.array(windows) + 2**5 - 1)
    assert np.ptp(full.num_transitions[:, 0]) > 0

    # On an independent Gaussian, Var(x)/Var(score) is sigma^4 for any cloud
    # with nonzero spread. Only the explicit shrinkage changes this estimate.
    truth = np.asarray(data["scale"][jobs.dataset]) ** 2
    metric = np.ones_like(truth)
    for count in full.num_transitions[:, :-1].T:
        weight = count[:, None] / (count[:, None] + 3)
        metric = np.exp((1 - weight) * np.log(metric) + weight * np.log(truth))
    np.testing.assert_allclose(full.jobs.inverse_mass, metric, rtol=1e-8)
    draws = sample(density, full.jobs, 4, data=data, num_lanes=3, max_depth=5)
    assert draws.position.shape == (len(jobs.position), 4, 5)
    assert np.all(np.isfinite(draws.position))


def curved(q):
    return -0.5 * (q[0] ** 2 + (q[1] - 0.6 * (q[0] ** 2 - 1)) ** 2)


def test_first_metric_matches_explicit_leaf_probabilities():
    keys = jax.random.split(jax.random.key(23), 5)
    jobs = Jobs(keys, np.tile([0.2, 0.1], (5, 1)), np.full(5, 0.4), np.ones((5, 2)))
    fit = warmup(curved, jobs, max_depth=3, window_evaluations=(1, 25))
    # A budget of one evaluates one entire tree. Explicitly keep its leaves and
    # multiply each leaf's probability through the subsequent subtree choices.
    work_keys = jax.vmap(jax.random.split)(keys)[:, 0]
    jobs = jobs._replace(key=jax.random.key_data(work_keys), dataset=np.zeros(5, np.int32))
    ld = lambda q, _: curved(q)
    lanes, chains = _empty_lanes(5, 2, jobs.position.dtype, 3, None)
    lanes, chains = _assign(ld, lanes, chains, np.ones(5, bool), np.arange(5), jobs, None)

    @jax.jit
    def observe(lane, chain):
        state, _, ended, sub_done, point, (_, p_tree) = _advance(ld, 3, chain, lane)
        energy = -curved(point.position) + jnp.dot(point.momentum, point.momentum) / 2
        return (
            _begin_next(chain, state, ended, sub_done),
            ended,
            sub_done,
            p_tree,
            jnp.concatenate((point.position, point.grad)),
            lane.energy - energy,
        )

    for i in range(5):
        lane, chain = jax.tree.map(lambda x, i=i: x[i], (lanes, chains))
        values = [np.r_[jobs.position[i], lane.proposal.grad]]
        weights = np.ones(1)
        leaves, logweights = [], []
        for _ in range(7):
            lane, ended, sub_done, p, leaf, logw = observe(lane, chain)
            leaves.append(np.asarray(leaf))
            logweights.append(float(logw))
            if sub_done:
                w = np.exp(logweights - np.max(logweights))
                weights = np.r_[(1 - p) * weights, p * w / w.sum()]
                values.extend(leaves)
                leaves, logweights = [], []
            if ended:
                break
        assert ended
        values = np.array(values)
        mean = weights @ values
        var = weights @ (values - mean) ** 2
        identified = (var[:2] > 0) & (var[2:] > 0)
        expected = np.ones(2)
        expected[identified] = (var[:2][identified] / var[2:][identified]) ** (1 / 8)
        np.testing.assert_allclose(fit.jobs.inverse_mass[i], expected, rtol=1e-10)
    assert np.max(abs(fit.jobs.inverse_mass - 1)) > 0.01


def test_dual_averaging_uses_mean_acceptance():
    initialize, update, _ = dual_averaging_adaptation(0.83)
    eps = jnp.array(0.06)
    reference, actual = initialize(eps), _start_dual(eps)
    log_average = 0.0
    for t, alpha in enumerate((1.0, 0.8, 0.2, 0.0, 0.99, 0.73), 1):
        reference = update(reference, alpha)
        actual, step = _dual(actual, alpha, 0.83)
        np.testing.assert_allclose(step, np.exp(reference.log_step_size), rtol=1e-6)
        # Hoffman and Gelman, Algorithm 6, averages the newly proposed log step.
        weight = t**-0.75
        log_average = weight * reference.log_step_size + (1 - weight) * log_average
        np.testing.assert_allclose(actual.log_average, log_average, rtol=1e-6)


def test_rejected_leaves_and_invalid_start():
    def constrained(x):
        return jnp.where(jnp.abs(x[0]) < 0.001, -(x[0] ** 2), -jnp.inf)

    jobs = Jobs(
        jax.random.split(jax.random.key(8), 1),
        np.zeros((1, 1), np.float32),
        np.ones(1, np.float32),
        np.ones((1, 1), np.float32),
    )
    fit = warmup(constrained, jobs, window_evaluations=(1, 1))
    np.testing.assert_array_equal(fit.num_divergences, [[1, 1]])
    np.testing.assert_array_equal(fit.jobs.inverse_mass, [[1.0]])
    assert np.isfinite(fit.jobs.step_size).all()
    with pytest.raises(ValueError, match="log density"):
        warmup(constrained, jobs._replace(position=np.ones_like(jobs.position)))
    with pytest.raises(ValueError, match="budgets"):
        warmup(constrained, jobs, window_evaluations=(0, 2))
