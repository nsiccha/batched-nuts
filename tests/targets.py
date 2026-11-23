"""Synthetic targets and chains for the tests."""

import jax
import jax.numpy as jnp

import batched_nuts


def gaussian(position, data):
    return -0.5 * jnp.sum(((position - data["mean"]) / data["scale"]) ** 2)


def gaussian_jobs(dim=50, seed=0):
    """Two Gaussians with scales from 0.1 to 3, and 16 chains that behave very differently.

    With the inverse mass set to the variances a step size below 2 is stable; with a unit
    mass the limit is 0.2.  The chains use ordinary step sizes, tiny ones (trajectories of
    the maximal length) and ones at and beyond the limit (divergences).
    """
    scale = jnp.exp(jnp.linspace(jnp.log(0.1), jnp.log(3.0), dim))
    data = dict(
        mean=jnp.stack([jnp.zeros(dim), jnp.linspace(-1, 1, dim)]),
        scale=jnp.stack([scale, scale[::-1]]),
    )
    v, u = "variance", "unit"  # the inverse mass: the variances or ones
    settings = [  # dataset, inverse mass, step size
        (0, v, 0.9), (0, v, 0.3), (0, v, 0.01), (0, u, 0.05),
        (0, u, 0.19), (0, u, 0.205), (0, v, 1.9), (0, v, 2.02),
        (1, v, 1.2), (1, u, 0.02), (1, v, 0.6), (1, u, 0.21),
        (1, v, 0.005), (1, v, 1.99), (1, u, 0.1), (1, v, 0.15),
    ]  # fmt: skip
    dataset = jnp.array([s[0] for s in settings])
    key, start_key = jax.random.split(jax.random.key(seed))
    noise = jax.random.normal(start_key, (len(settings), dim))
    jobs = batched_nuts.Jobs(
        key=jax.random.split(key, len(settings)),
        position=data["mean"][dataset] + data["scale"][dataset] * noise,
        step_size=jnp.array([s[2] for s in settings]),
        inverse_mass=jnp.stack(
            [data["scale"][s[0]] ** 2 if s[1] == v else jnp.ones(dim) for s in settings]
        ),
        dataset=dataset,
    )
    return gaussian, jobs, data


def funnel(position):
    """Neal's funnel: v ~ normal(0, 3), x[i] ~ normal(0, exp(v / 2))."""
    v, x = position[0], position[1:]
    return -(v**2) / 18 - 0.5 * jnp.sum(x**2) * jnp.exp(-v) - 0.5 * x.size * v


def funnel_jobs(dim=10, num_chains=12, seed=1):
    """Chains on the funnel with step sizes from 0.03 to 0.6: all trajectory lengths."""
    key, start_key = jax.random.split(jax.random.key(seed))
    return (
        funnel,
        batched_nuts.Jobs(
            key=jax.random.split(key, num_chains),
            position=jax.random.normal(start_key, (num_chains, dim)),
            step_size=jnp.exp(jnp.linspace(jnp.log(0.03), jnp.log(0.6), num_chains)),
            inverse_mass=jnp.ones((num_chains, dim)),
        ),
        None,
    )


def logistic_regression(position, data):
    logits = data["x"] @ position
    return jnp.sum(data["y"] * logits - jnp.logaddexp(0.0, logits)) - 0.5 * jnp.sum(
        position**2
    )


def logistic_regression_jobs(num_datasets=3, num_chains=4, num_rows=200, dim=5, seed=2):
    """Simulated logistic regressions, `num_chains` chains each, datasets interleaved."""
    key, x_key, coef_key, y_key, start_key = jax.random.split(jax.random.key(seed), 5)
    x = jax.random.normal(x_key, (num_datasets, num_rows, dim))
    coef = jax.random.normal(coef_key, (num_datasets, dim))
    y = jax.random.bernoulli(y_key, jax.nn.sigmoid(jnp.einsum("knd,kd->kn", x, coef)))
    num_jobs = num_datasets * num_chains
    return (
        logistic_regression,
        batched_nuts.Jobs(
            key=jax.random.split(key, num_jobs),
            position=jax.random.normal(start_key, (num_jobs, dim)),
            step_size=jnp.exp(jnp.linspace(jnp.log(0.02), jnp.log(0.4), num_jobs)),
            inverse_mass=jnp.full((num_jobs, dim), 0.05),
            dataset=jnp.arange(num_jobs) % num_datasets,
        ),
        dict(x=x, y=y.astype(float)),
    )
