"""The convergence diagnostics of retained draws.

The reference values are analytic where they can be.  Independent draws have no
autocorrelation, so their effective size is the number of draws; an AR(1) chain of
correlation `rho` has the effective size `num_draws * (1 - rho) / (1 + rho)`; and rank
normalization is invariant under a monotone transformation, which the variance-based
R-hat is not.  The rest of the file pins the failure modes the diagnostics exist for.
"""

import numpy as np
import pytest

import batched_nuts
from batched_nuts.diagnostics import ess, rank_normalize, split_rhat, summary


def ar1(num_chains, num_draws, rho, seed=0):
    """Independent AR(1) chains of correlation `rho`, each with unit marginal variance."""
    noise = np.random.default_rng(seed).normal(size=(num_chains, num_draws))
    draws = np.zeros((num_chains, num_draws))
    for t in range(1, num_draws):
        draws[:, t] = rho * draws[:, t - 1] + np.sqrt(1 - rho**2) * noise[:, t]
    return draws[..., None]


def test_independent_chains_look_converged():
    draws = np.random.default_rng(1).normal(size=(8, 1000, 3))
    total = draws.shape[0] * draws.shape[1]
    assert np.all(split_rhat(draws) < 1.01)
    # Without autocorrelation the size is the number of draws, up to the noise of the
    # estimate itself.
    assert np.all(ess(draws, kind="bulk") > 0.9 * total)
    assert np.all(ess(draws, kind="tail") > 0.85 * total)
    assert summary(draws).converged(min_ess=1000)


def test_a_chain_that_has_not_mixed_is_flagged():
    draws = np.random.default_rng(2).normal(size=(4, 500, 2))
    draws[0] += 2.0
    total = draws.shape[0] * draws.shape[1]
    assert np.all(split_rhat(draws) > 1.2)
    # Between-chain variance dominates, so the draws count for very little.
    assert np.all(ess(draws, kind="bulk") < 0.05 * total)
    assert not summary(draws).converged()


def test_autocorrelated_chains_have_the_analytic_size():
    rho = 0.9
    draws = ar1(8, 4000, rho)
    analytic = draws.shape[0] * draws.shape[1] * (1 - rho) / (1 + rho)
    assert np.all(split_rhat(draws) < 1.02)
    assert abs(float(ess(draws, kind="bulk")[0]) / analytic - 1) < 0.2, analytic
    # The estimate is not the nominal count: that is the whole point of the diagnostic.
    assert float(ess(draws, kind="bulk")[0]) < 0.3 * draws.shape[0] * draws.shape[1]


def test_the_tail_size_notices_a_chain_with_the_wrong_tail():
    draws = np.random.default_rng(3).normal(size=(8, 2000, 1))
    draws[0, -300:, 0] += 6.0
    bulk = float(ess(draws, kind="bulk")[0])
    tail = float(ess(draws, kind="tail")[0])
    # The bulk of a chain that is mostly right keeps its size; the tail does not.
    assert tail < bulk / 3, (bulk, tail)


def test_rank_normalization_ignores_monotone_transformations():
    draws = np.random.default_rng(4).normal(size=(4, 300, 2))
    # The ranks of exp(x) are those of x, so the rank-normalized R-hat is unchanged.
    # A variance-based R-hat would report a wildly different number here.
    np.testing.assert_array_equal(split_rhat(np.exp(draws)), split_rhat(draws))


def test_draws_that_do_not_vary_are_reported_as_nan():
    draws = np.full((3, 40, 2), 2.5)
    assert np.all(np.isnan(split_rhat(draws)))
    assert np.all(np.isnan(ess(draws, kind="bulk")))
    assert not summary(draws).converged()


def test_tied_draws_are_ordered_by_their_average_rank():
    draws = np.round(np.random.default_rng(5).normal(size=(4, 200, 2)) * 2) / 2
    assert len(np.unique(draws)) < draws.size  # the draws really do repeat
    assert np.all(split_rhat(draws) < 1.02)
    assert np.all(ess(draws, kind="bulk") <= draws.shape[0] * draws.shape[1])


def test_an_odd_number_of_draws_uses_both_halves():
    draws = np.random.default_rng(6).normal(size=(4, 301, 2))
    np.testing.assert_array_equal(split_rhat(draws), split_rhat(draws[:, :300]))
    np.testing.assert_array_equal(ess(draws, kind="bulk"), ess(draws[:, :300], kind="bulk"))


def test_rank_normalization_keeps_the_shape_of_one_chain():
    draws = np.random.default_rng(7).normal(size=(1, 50, 3))
    normalized = rank_normalize(draws)
    assert normalized.shape == draws.shape
    # A single chain is still ranked against itself.
    assert np.all(np.argsort(normalized[0, :, 0]) == np.argsort(draws[0, :, 0]))


def test_shapes_and_arguments_are_checked():
    with pytest.raises(ValueError, match="shape"):
        split_rhat(np.zeros(10))
    with pytest.raises(ValueError, match="at least 2 chains"):
        split_rhat(np.zeros((1, 10, 2)))
    with pytest.raises(ValueError, match="at least 4 draws"):
        split_rhat(np.zeros((3, 3, 2)))
    with pytest.raises(ValueError, match="'bulk' or 'tail'"):
        ess(np.zeros((3, 10, 2)), kind="tail_area")


def test_summary_accepts_the_shapes_a_sampler_returns():
    draws = np.random.default_rng(8).normal(size=(4, 400))
    flat = summary(draws)
    assert flat.rhat.shape == (1,) and flat.mean.shape == (1,)
    structured = summary(draws[..., None])
    np.testing.assert_array_equal(flat.rhat, structured.rhat)
    np.testing.assert_allclose(flat.sd, structured.sd)


def test_summary_reads_the_draws_of_the_sampler():
    """The diagnostics of a real run, on a Gaussian every chain has already found."""
    import jax
    import jax.numpy as jnp

    dim, num_chains, num_draws = 5, 8, 120
    jobs = batched_nuts.Jobs(
        key=jax.random.split(jax.random.key(11), num_chains),
        position=jax.random.normal(jax.random.key(12), (num_chains, dim)),
        step_size=jnp.full(num_chains, 0.4),
        inverse_mass=jnp.ones((num_chains, dim)),
    )
    draws = batched_nuts.sample(
        lambda x: -0.5 * jnp.sum(x**2), jobs, num_draws, max_depth=6
    )
    result = batched_nuts.summary(draws)
    np.testing.assert_array_equal(result.rhat, split_rhat(draws.position))
    assert np.all(result.rhat < 1.05)
    assert np.all(result.ess_bulk > 50)
    assert np.all(np.abs(result.mean) < 0.5)
    assert np.all(result.sd > 0.5)
