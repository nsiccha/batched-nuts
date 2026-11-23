"""The draws of a chain do not depend on the lane it ran in, nor on when it ran."""

import numpy as np
import pytest

import batched_nuts

from targets import gaussian_jobs

NUM_DRAWS = 12


@pytest.fixture(scope="module")
def all_at_once():
    logdensity, jobs, data = gaussian_jobs()
    return batched_nuts.sample(logdensity, jobs, NUM_DRAWS, data=data, max_depth=7)


@pytest.mark.parametrize("num_lanes, buffer_size", [(5, None), (1, None), (7, 7), (16, 20)])
def test_lanes_and_buffer_do_not_change_draws(all_at_once, num_lanes, buffer_size):
    logdensity, jobs, data = gaussian_jobs()
    draws = batched_nuts.sample(
        logdensity,
        jobs,
        NUM_DRAWS,
        data=data,
        num_lanes=num_lanes,
        max_depth=7,
        buffer_size=buffer_size,
    )
    np.testing.assert_array_equal(draws.position, all_at_once.position)
    for name in ["num_integration_steps", "is_divergent", "is_turning"]:
        np.testing.assert_array_equal(getattr(draws, name), getattr(all_at_once, name))

    # every gradient that advanced a chain is a leapfrog step of a recorded transition
    assert draws.num_evaluations == draws.num_integration_steps.sum()
    assert draws.num_evaluations <= draws.num_iterations * num_lanes
    if num_lanes == 1:  # a single lane with a queue behind it never idles
        assert draws.num_evaluations == draws.num_iterations


def test_fewer_lanes_waste_fewer_gradients(all_at_once):
    """A lane without a queue behind it idles once its chain is done."""
    logdensity, jobs, data = gaussian_jobs()
    queued = batched_nuts.sample(
        logdensity, jobs, NUM_DRAWS, data=data, num_lanes=4, max_depth=7
    )
    useful = lambda draws, lanes: draws.num_evaluations / (draws.num_iterations * lanes)
    assert useful(queued, 4) > useful(all_at_once, 16)
