"""Given the same keys, every transition is the transition of `blackjax.nuts`.

The chains of each test behave very differently (short and maximal trajectories, U-turns,
divergences), and there are fewer lanes than chains, so that lanes take over new chains
and, where the chains have data, new datasets.
"""

import numpy as np
import pytest

import batched_nuts

from blackjax_baseline import make_blackjax_sampler
from targets import funnel_jobs, gaussian_jobs, logistic_regression_jobs

NUM_DRAWS = 25


def run_both(make_jobs, num_lanes, max_depth):
    logdensity, jobs, data = make_jobs()
    draws = batched_nuts.sample(
        logdensity, jobs, NUM_DRAWS, data=data, num_lanes=num_lanes, max_depth=max_depth
    )
    reference = make_blackjax_sampler(logdensity, max_depth)(jobs, NUM_DRAWS, data)
    reference = {name: np.asarray(value) for name, value in reference.items()}

    # the comparison only means something if the chains took different paths
    steps = draws.num_integration_steps
    assert len(np.unique(steps)) >= 6
    assert np.any(steps == 2**max_depth - 1)
    assert np.any(draws.is_turning) and np.any(~draws.is_turning)
    return draws, reference


def assert_same_trajectories(draws, reference):
    for name in ["num_integration_steps", "is_divergent", "is_turning"]:
        np.testing.assert_array_equal(getattr(draws, name), reference[name], err_msg=name)


def test_same_draws_bit_for_bit():
    """Independent normal coordinates: all arithmetic that decides a draw is elementwise."""
    draws, reference = run_both(gaussian_jobs, num_lanes=6, max_depth=7)
    assert np.any(draws.is_divergent)
    assert_same_trajectories(draws, reference)
    np.testing.assert_array_equal(draws.position, reference["position"])
    # sums over the coordinates may be rounded differently in the two programs
    for name in ["logdensity", "acceptance_rate"]:
        np.testing.assert_allclose(
            getattr(draws, name), reference[name], rtol=1e-12, atol=0
        )


@pytest.mark.parametrize(
    "make_jobs, num_lanes, max_depth",
    [(funnel_jobs, 5, 8), (logistic_regression_jobs, 5, 6)],
)
def test_same_draws_up_to_rounding(make_jobs, num_lanes, max_depth):
    """Gradients that contain sums: the two programs may round a sum differently.

    A difference in the last bit of a gradient grows along the chain, so the draws are
    compared with a tolerance, over few transitions.  The trajectories are still the
    same: equal numbers of leapfrog steps and equal reasons to stop, every time.
    """
    draws, reference = run_both(make_jobs, num_lanes, max_depth)
    assert_same_trajectories(draws, reference)
    for name in ["position", "logdensity", "acceptance_rate"]:
        np.testing.assert_allclose(
            getattr(draws, name), reference[name], rtol=1e-8, atol=1e-10, err_msg=name
        )
