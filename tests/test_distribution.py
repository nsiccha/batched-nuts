"""The draws have the moments of the target.

Every chain starts from an exact draw of the target, so all of its draws are distributed
as the target and its averages are unbiased.  The chains are independent, so the standard
error of the mean over chains follows from the spread between chains, without any
estimate of autocorrelation.
"""

import jax
import jax.numpy as jnp
import numpy as np

import batched_nuts

NUM_CHAINS, NUM_DRAWS = 64, 150


def z_scores(per_chain, truth):
    """(mean over chains - truth) / standard error, for per-chain averages."""
    standard_error = per_chain.std(0, ddof=1) / np.sqrt(len(per_chain))
    return (per_chain.mean(0) - truth) / standard_error


def test_scaled_normal():
    """Scales from 0.1 to 3 with a unit mass matrix: long trajectories of varying length."""
    dim = 20
    scale = jnp.exp(jnp.linspace(jnp.log(0.1), jnp.log(3.0), dim))
    key, start_key = jax.random.split(jax.random.key(3))
    jobs = batched_nuts.Jobs(
        key=jax.random.split(key, NUM_CHAINS),
        position=scale * jax.random.normal(start_key, (NUM_CHAINS, dim)),
        step_size=jnp.full(NUM_CHAINS, 0.08),
        inverse_mass=jnp.ones((NUM_CHAINS, dim)),
    )
    draws = batched_nuts.sample(
        lambda x: -0.5 * jnp.sum((x / scale) ** 2),
        jobs,
        NUM_DRAWS,
        num_lanes=24,
        max_depth=8,
    )
    assert not draws.is_divergent.any()
    assert len(np.unique(draws.num_integration_steps)) >= 4
    x = draws.position / np.asarray(scale)
    z = np.concatenate([z_scores(x.mean(1), 0.0), z_scores((x**2).mean(1), 1.0)])
    # 40 statistics: |z| > 4.5 has probability 3e-4 under the null
    assert np.max(np.abs(z)) < 4.5, z


def test_correlated_normal():
    """Correlation 0.95 in two dimensions."""
    covariance = jnp.array([[1.0, 0.95], [0.95, 1.0]])
    precision = jnp.linalg.inv(covariance)
    key, start_key = jax.random.split(jax.random.key(4))
    jobs = batched_nuts.Jobs(
        key=jax.random.split(key, NUM_CHAINS),
        position=jax.random.multivariate_normal(
            start_key, jnp.zeros(2), covariance, (NUM_CHAINS,)
        ),
        step_size=jnp.full(NUM_CHAINS, 0.15),
        inverse_mass=jnp.ones((NUM_CHAINS, 2)),
    )
    draws = batched_nuts.sample(lambda x: -0.5 * x @ precision @ x, jobs, NUM_DRAWS)
    x = draws.position
    z = np.concatenate(
        [
            z_scores(x.mean(1), 0.0),
            z_scores((x**2).mean(1), 1.0),
            z_scores((x[..., 0] * x[..., 1]).mean(1)[:, None], 0.95),
        ]
    )
    assert np.max(np.abs(z)) < 4.0, z
