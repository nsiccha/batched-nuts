"""Starts for hierarchical models: the population marginal, then conditional effects."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from batched_nuts import Jobs, hierarchical_start, sample


def two_level(num_groups=12, per_group=5, seed=0):
    """Intercepts and slopes per group with known scales, so the posterior is normal."""
    x_key, u_key, v_key, noise_key = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(x_key, (num_groups, per_group))
    u = jax.random.normal(u_key, (num_groups, 1))
    v = jax.random.normal(v_key, (num_groups, 1))
    y = 1.0 + 0.7 * u + (-0.5 + 0.4 * v) * x + 0.6 * jax.random.normal(noise_key, x.shape)

    def logdensity(position):
        a, b = position[0], position[1]
        u = position[2 : 2 + num_groups, None]
        v = position[2 + num_groups :, None]
        mean = a + 0.7 * u + (b + 0.4 * v) * x
        return (
            -0.5 * jnp.sum(((y - mean) / 0.6) ** 2)
            - 0.5 * jnp.sum(u**2)
            - 0.5 * jnp.sum(v**2)
            - 0.005 * (a**2 + b**2)
        )

    index = np.arange(num_groups)
    return logdensity, np.arange(2), np.stack([2 + index, 2 + num_groups + index], 1)


def scaled_groups_data(num_groups=8, seed=1):
    """Simulated group estimates and their known standard errors."""
    effect_key, noise_key = jax.random.split(jax.random.key(seed))
    error = jnp.linspace(9.0, 18.0, num_groups)
    y = (
        4.0
        + 6.0 * jax.random.normal(effect_key, (num_groups,))
        + error * jax.random.normal(noise_key, (num_groups,))
    )
    return y, error


def scaled_groups(centered, num_groups=8, seed=1):
    """One effect per group with a group scale to estimate, in either parametrization."""
    y, error = scaled_groups_data(num_groups, seed)

    def logdensity(position):
        mu, log_tau, effect = position[0], position[1], position[2:]
        tau = jnp.exp(log_tau)
        if centered:
            theta = effect
            prior = -0.5 * jnp.sum(((effect - mu) / tau) ** 2) - num_groups * log_tau
        else:
            theta = mu + tau * effect
            prior = -0.5 * jnp.sum(effect**2)
        scale_prior = -jnp.log1p((tau / 5.0) ** 2) + log_tau
        return (
            -0.5 * jnp.sum(((y - theta) / error) ** 2)
            + prior
            - 0.5 * (mu / 5.0) ** 2
            + scale_prior
        )

    return logdensity, np.arange(2), 2 + np.arange(num_groups)[:, None]


def decay(num_groups=6, seed=3):
    """Exponential decay with an amplitude and a rate effect per group: not normal."""
    time = jnp.linspace(0.5, 8.0, 6)
    a_key, k_key, noise_key = jax.random.split(jax.random.key(seed), 3)
    amplitude = jnp.exp(1.0 + 0.3 * jax.random.normal(a_key, (num_groups, 1)))
    rate = jnp.exp(-1.0 + 0.2 * jax.random.normal(k_key, (num_groups, 1)))
    y = amplitude * jnp.exp(-rate * time)
    y = y + 0.1 * jax.random.normal(noise_key, y.shape)

    def logdensity(position):
        log_a, log_k, log_sa, log_sk, log_sy = position[:5]
        ea = position[5 : 5 + num_groups, None]
        ek = position[5 + num_groups :, None]
        rate = jnp.exp(log_k + jnp.exp(log_sk) * ek)
        mean = jnp.exp(log_a + jnp.exp(log_sa) * ea - rate * time)
        return (
            -0.5 * jnp.sum(((y - mean) / jnp.exp(log_sy)) ** 2)
            - y.size * log_sy
            - 0.5 * jnp.sum(ea**2)
            - 0.5 * jnp.sum(ek**2)
            - 0.125 * jnp.sum(position[:5] ** 2)
        )

    index = np.arange(num_groups)
    return logdensity, np.arange(5), np.stack([5 + index, 5 + num_groups + index], 1)


def test_normal_posterior_is_drawn_exactly():
    logdensity, population, groups = two_level()
    dim = population.size + groups.size
    precision = -np.asarray(jax.hessian(logdensity)(jnp.zeros(dim)))
    mean = np.linalg.solve(precision, np.asarray(jax.grad(logdensity)(jnp.zeros(dim))))
    covariance = np.linalg.inv(precision)

    start = hierarchical_start(logdensity, population, groups, 4000, jax.random.key(1))

    assert start.num_converged == start.num_agreeing == len(start.num_steps) == 8
    error = np.sqrt(np.diag(covariance)[:2])
    assert np.max(np.abs(start.population - mean[:2]) / error) < 1e-4
    np.testing.assert_allclose(start.covariance, covariance[:2, :2], rtol=1e-6)
    np.testing.assert_allclose(start.inverse_mass[0], np.diag(covariance), rtol=1e-6)
    np.testing.assert_array_equal(
        start.inverse_mass, start.inverse_mass[:1].repeat(4000, 0)
    )
    # Whitened by the exact posterior, the starts are standard normal in every direction.
    white = (start.position - mean) @ np.linalg.cholesky(precision)
    assert np.max(np.abs(white.mean(0))) < 5 / np.sqrt(4000)
    np.testing.assert_allclose(np.cov(white.T), np.eye(dim), atol=0.12)


def test_parametrization_of_the_effects_does_not_matter():
    key = jax.random.key(2)
    centered = hierarchical_start(*scaled_groups(True), 64, key)
    scaled = hierarchical_start(*scaled_groups(False), 64, key)

    for start in (centered, scaled):
        assert start.num_converged == start.num_agreeing == 8
    np.testing.assert_allclose(scaled.population, centered.population, atol=1e-6)
    np.testing.assert_allclose(scaled.covariance, centered.covariance, rtol=1e-5)
    np.testing.assert_allclose(scaled.position[:, :2], centered.position[:, :2], atol=1e-6)
    mu, tau = scaled.position[:, :1], np.exp(scaled.position[:, 1:2])
    np.testing.assert_allclose(
        mu + tau * scaled.position[:, 2:], centered.position[:, 2:], rtol=1e-5, atol=1e-6
    )
    # The scale is uncertain with 8 groups: the starts must spread over it.
    assert centered.position[:, 1].std() > 0.3


def test_slice_update_follows_a_marginal_that_is_not_normal():
    logdensity, population, groups = scaled_groups(False)
    y, error = (np.asarray(a) for a in scaled_groups_data())
    # The exact marginal of (mu, log tau) on a grid, with the effects integrated out
    # in closed form: y[j] is normal around mu with variance error[j]**2 + tau**2.
    mu = np.linspace(-25.0, 35.0, 601)[:, None]
    log_tau = np.linspace(-9.0, 6.0, 901)[None, :]
    variance = error**2 + np.exp(2 * log_tau[..., None])
    density = (
        -0.5 * ((y - mu[..., None]) ** 2 / variance).sum(-1)
        - 0.5 * np.log(variance).sum(-1)
        - 0.5 * (mu / 5.0) ** 2
        - np.log1p(np.exp(2 * log_tau) / 25.0)
        + log_tau
    )
    weight = np.exp(density - density.max())
    weight /= weight.sum()
    mean = (weight * log_tau).sum()
    scale = np.sqrt((weight * (log_tau - mean) ** 2).sum())

    def scale_error(num_sweeps):
        start = hierarchical_start(
            logdensity, population, groups, 4000, jax.random.key(8), num_sweeps=num_sweeps
        )
        draws = start.position[:, 1]
        low = np.mean(draws < mean - 1.645 * scale)
        return (draws.mean() - mean) / scale, draws.var() / scale**2, low

    bias, ratio, low = scale_error(1)
    assert abs(bias) < 0.1 and 0.85 < ratio < 1.15 and 0.03 < low < 0.08
    # With 8 groups the normal draw alone misses the lower tail of the group scale.
    normal_bias, normal_ratio, normal_low = scale_error(0)
    assert normal_bias > 3 * abs(bias) and normal_ratio < 0.7 and normal_low < 0.01


def test_nonlinear_effects_match_an_independent_laplace_fit():
    logdensity, population, groups = decay()
    num_effects = groups.size
    start = hierarchical_start(logdensity, population, groups, 32, jax.random.key(4))
    assert start.num_converged >= 1 and np.all(np.isfinite(start.position))

    negative = jax.jit(lambda theta, e: -logdensity(jnp.concatenate([theta, e])))
    gradient = jax.jit(jax.grad(negative, argnums=1))
    hessian = jax.jit(jax.hessian(negative, argnums=1))

    def laplace(theta):
        """Minus the Laplace marginal with a generic optimizer and a dense Hessian."""
        theta = jnp.asarray(theta)
        fit = minimize(
            lambda e: float(negative(theta, jnp.asarray(e))),
            np.zeros(num_effects),
            jac=lambda e: np.asarray(gradient(theta, jnp.asarray(e))),
            method="BFGS",
            options=dict(gtol=1e-9),
        )
        curvature = np.asarray(hessian(theta, jnp.asarray(fit.x)))
        return fit.fun + 0.5 * np.linalg.slogdet(curvature)[1]

    error = np.sqrt(np.diag(start.covariance))
    center = laplace(start.population)
    for i in range(population.size):
        step = np.zeros(population.size)
        step[i] = 0.05 * error[i]
        up, down = laplace(start.population + step), laplace(start.population - step)
        # The fit is a stationary point of the independent objective, with its curvature.
        assert abs(up - down) / (2 * 0.05) < 2e-3
        curvature = (up - 2 * center + down) / step[i] ** 2
        assert curvature == pytest.approx(np.linalg.inv(start.covariance)[i, i], rel=0.02)
    # One mixed second difference checks a curvature between two parameters.
    first, second = 0.05 * error[0] * np.eye(5)[0], 0.05 * error[1] * np.eye(5)[1]
    corners = [
        laplace(start.population + a * first + b * second)
        for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ]
    mixed = (corners[0] - corners[1] - corners[2] + corners[3]) / (
        0.01 * error[0] * error[1]
    )
    precision = np.linalg.inv(start.covariance)
    assert mixed == pytest.approx(
        precision[0, 1], abs=0.02 * np.sqrt(precision[0, 0] * precision[1, 1])
    )


def test_sampler_takes_the_starts():
    logdensity, population, groups = two_level()
    dim = population.size + groups.size
    start = hierarchical_start(logdensity, population, groups, 32, jax.random.key(5))
    jobs = Jobs(
        key=jax.random.split(jax.random.key(6), 32),
        position=start.position,
        step_size=np.full(32, 0.4),
        inverse_mass=start.inverse_mass,
    )
    draws = sample(logdensity, jobs, num_draws=40)
    assert draws.position.shape == (32, 40, dim)
    assert not draws.is_divergent.any()
    # Chains that start as posterior draws stay where they started.
    scale = np.sqrt(start.inverse_mass[0])
    drift = (draws.position.mean((0, 1)) - start.position.mean(0)) / scale
    assert np.max(np.abs(drift)) < 1.0


def test_data_argument_matches_a_closure():
    logdensity, population, groups = two_level()

    def with_data(position, data):
        return logdensity(position) + 0.0 * data["offset"]

    key = jax.random.key(9)
    plain = hierarchical_start(logdensity, population, groups, 16, key)
    passed = hierarchical_start(
        with_data, population, groups, 16, key, data=dict(offset=jnp.zeros(()))
    )
    np.testing.assert_allclose(passed.population, plain.population, atol=1e-9)
    np.testing.assert_allclose(passed.position, plain.position, atol=1e-7)


def test_fits_that_end_elsewhere_are_reported():
    y = jnp.array([3.6, 4.4, 4.1, 3.9])

    def logdensity(position):  # the marginal of position[0] has two modes, at +2 and -2
        a, u = position[0], position[1:]
        return -0.5 * jnp.sum((y - a**2 - u) ** 2) - 0.5 * jnp.sum(u**2) - 0.005 * a**2

    with pytest.warns(UserWarning, match="ended elsewhere"):
        start = hierarchical_start(
            logdensity, np.arange(1), 1 + np.arange(4)[:, None], 8, jax.random.key(10)
        )
    assert start.num_agreeing < start.num_converged == 8
    assert abs(abs(start.population[0]) - 2.0) < 0.1

    # A fit that cannot start is left out and the others are used.
    initial = np.linspace(0.5, 1.9, 8)[:, None]
    initial[3] = np.nan
    start = hierarchical_start(
        logdensity,
        np.arange(1),
        1 + np.arange(4)[:, None],
        8,
        jax.random.key(10),
        initial=initial,
    )
    assert start.num_converged == start.num_agreeing == 7


def test_models_without_a_proper_fit_are_refused():
    y = jnp.array([3.0, 4.0, 2.5, 3.5, 5.0])
    key = jax.random.key(11)

    def undetermined(position):  # only 0.1 a + 0.9 b enters, and neither has a prior
        a, b, u = position[0], position[1], position[2:]
        return -0.5 * jnp.sum((y - 0.1 * a - 0.9 * b - u) ** 2) - 0.5 * jnp.sum(u**2)

    with pytest.raises(RuntimeError, match="none of the 8 fits"):
        hierarchical_start(undetermined, np.arange(2), 2 + np.arange(5)[:, None], 4, key)

    def saddle(position):  # zero effects are a stationary point that is not a mode
        a, u, v = position[0], position[1:6], position[6:]
        mean = a + u**2 + v**2
        return -2.0 * jnp.sum((y - mean) ** 2) - 0.5 * jnp.sum(u**2 + v**2) - 50.0 * a**2

    groups = np.stack([1 + np.arange(5), 6 + np.arange(5)], 1)
    with pytest.raises(RuntimeError, match="none of the 8 fits"):
        hierarchical_start(saddle, np.arange(1), groups, 4, key)


def test_wrong_inputs_are_refused():
    logdensity, population, groups = two_level()
    key = jax.random.key(7)
    with pytest.raises(ValueError, match="exactly once"):
        hierarchical_start(logdensity, population, groups[:-1], 4, key)
    with pytest.raises(ValueError, match="shape"):
        hierarchical_start(logdensity, population[:, None], groups, 4, key)
    with pytest.raises(ValueError, match="initial"):
        hierarchical_start(logdensity, population, groups, 4, key, initial=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="num_sweeps"):
        hierarchical_start(logdensity, population, groups, 4, key, num_sweeps=-1)
    with pytest.raises(RuntimeError, match="none of the 8 fits"):
        hierarchical_start(lambda x: jnp.nan * jnp.sum(x), population, groups, 4, key)
