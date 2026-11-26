"""Impossible arguments are refused, by all three entry points, before any work starts.

A run in an unattended simulation study must not read a plausible-looking number out of a
call it could not honour.  `sample` used to accept every one of these: zero lanes spun in
the batched loop forever, zero draws failed on an `IndexError` in the last line of the
sampling loop, and `max_depth=0` died inside JAX's tracing.
"""

import jax
import jax.numpy as jnp
import pytest

import batched_nuts


def logdensity(x):
    return -0.5 * jnp.sum(x**2)


def jobs(num_chains=3, dim=2):
    return batched_nuts.Jobs(
        key=jax.random.split(jax.random.key(0), num_chains),
        position=jnp.zeros((num_chains, dim)),
        step_size=jnp.full(num_chains, 0.5),
        inverse_mass=jnp.ones((num_chains, dim)),
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"num_lanes": 0}, "num_lanes"),
        ({"num_lanes": -2}, "num_lanes"),
        ({"buffer_size": 0}, "buffer_size"),
        ({"buffer_size": -7}, "buffer_size"),
        ({"max_depth": 0}, "max_depth must be between 1 and 30"),
        ({"max_depth": 31}, "max_depth must be between 1 and 30"),
    ],
)
def test_sample_refuses_impossible_settings(arguments, message):
    with pytest.raises(ValueError, match=message):
        batched_nuts.sample(logdensity, jobs(), 4, **arguments)


@pytest.mark.parametrize("num_draws", [0, -3])
def test_sample_refuses_a_draw_count_that_cannot_be_honoured(num_draws):
    with pytest.raises(ValueError, match="num_draws must be a positive integer"):
        batched_nuts.sample(logdensity, jobs(), num_draws)


def test_sample_refuses_a_job_array_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"jobs\.step_size must have shape \(3,\)"):
        batched_nuts.sample(logdensity, jobs()._replace(step_size=jnp.ones(2)), 4)
    with pytest.raises(ValueError, match=r"jobs\.inverse_mass must have shape \(3, 2\)"):
        batched_nuts.sample(logdensity, jobs()._replace(inverse_mass=jnp.ones((3, 3))), 4)


def test_sample_refuses_a_position_that_is_not_a_table_of_chains():
    with pytest.raises(ValueError, match=r"\(num_chains, dim\)"):
        batched_nuts.sample(logdensity, jobs()._replace(position=jnp.zeros(3)), 4)


def test_every_entry_point_refuses_an_empty_queue():
    empty = batched_nuts.Jobs(
        key=jax.random.split(jax.random.key(0), 0),
        position=jnp.zeros((0, 2)),
        step_size=jnp.zeros(0),
        inverse_mass=jnp.zeros((0, 2)),
    )
    with pytest.raises(ValueError, match="requires a chain"):
        batched_nuts.sample(logdensity, empty, 4)
    with pytest.raises(ValueError, match="requires a chain"):
        batched_nuts.warmup(logdensity, empty)
    with pytest.raises(ValueError, match="requires a chain"):
        next(iter(batched_nuts.sample_batches(logdensity, empty)))


def test_every_entry_point_refuses_zero_lanes():
    with pytest.raises(ValueError, match="num_lanes"):
        batched_nuts.sample(logdensity, jobs(), 4, num_lanes=0)
    with pytest.raises(ValueError, match="requires a chain"):
        batched_nuts.warmup(logdensity, jobs(), num_lanes=0)
