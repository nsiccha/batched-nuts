"""The population-only Jacobian: the same derivatives, linear in the dimension."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from population_jacobian import population_jacobian


def effect_scales(position, population, weights, offsets):
    """One log scale per effect: nonlinear in the population, and in its own effect."""
    effects = np.setdiff1d(np.arange(position.shape[0]), population)
    values = position[population]
    return (
        offsets
        + jnp.tanh(weights @ values)
        + 0.1 * jnp.sum(values**2)
        + 0.3 * jnp.sin(position[effects])
    )


def problem(dimension, population, seed=0):
    rng = np.random.default_rng(seed + dimension)
    population = np.asarray(population)
    effects = dimension - len(population)
    return (
        population,
        jnp.asarray(rng.normal(size=dimension)),
        jnp.asarray(rng.normal(size=(effects, len(population)))),
        jnp.asarray(rng.normal(size=effects)),
    )


def two_large_axes(function, *arguments, dimension):
    """Shapes in the optimized program with two axes of at least half the dimension."""
    compiled = jax.jit(function).lower(*arguments).compile()
    shapes = set(re.findall(r"(?:f64|f32|s32|s64|pred)\[([0-9,]+)\]", compiled.as_text()))
    large = sorted(
        shape
        for shape in shapes
        if sum(int(n) >= dimension // 2 for n in shape.split(",")) >= 2
    )
    return large, compiled.memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize(
    "dimension,population",
    [(7, [0]), (12, [11, 0, 5]), (40, [3, 1, 2, 39, 20]), (935, list(range(926, 935)))],
)
def test_matches_population_columns_of_the_full_jacobian(dimension, population):
    population, position, weights, offsets = problem(dimension, population)

    def function(x):
        return effect_scales(x, population, weights, offsets)

    expected = jax.jacfwd(function)(position)[..., population]
    actual = population_jacobian(function, position, population)
    assert actual.shape == (dimension - len(population), len(population))
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    # The delta-method SD the initializer forms from it, inside a compiled function.
    covariance = 0.2 * (weights.T @ weights / len(weights) + jnp.eye(len(population)))

    def spread(jacobian):
        return jnp.sqrt(jnp.einsum("...i,ij,...j->...", jacobian, covariance, jacobian))

    compiled = jax.jit(lambda x: spread(population_jacobian(function, x, population)))
    np.testing.assert_allclose(compiled(position), spread(expected), rtol=1e-13, atol=1e-15)


def test_keeps_leading_output_axes():
    # Declared targets return one scale per effect family; effects index it as (J, b).
    population, position, weights, offsets = problem(30, [29, 4, 17])

    def function(x):
        scale = effect_scales(x, population, weights, offsets)
        return jnp.stack([scale[:9], jnp.exp(scale[9:18]), scale[18:] ** 2], axis=1)

    expected = jax.jacfwd(function)(position)[..., population]
    actual = population_jacobian(function, position, population)
    assert actual.shape == (9, 3, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_compiled_arrays_grow_linearly_with_the_dimension():
    temporary = {}
    for dimension in (935, 9_350, 93_500):
        population, position, weights, offsets = problem(
            dimension, np.arange(dimension - 9, dimension)
        )

        def selected(x, w, o, population=population):
            return population_jacobian(
                lambda y: effect_scales(y, population, w, o), x, population
            )

        large, temporary[dimension] = two_large_axes(
            selected, position, weights, offsets, dimension=dimension
        )
        assert large == []
    # Two tenfold dimension increases must stay far below quadratic growth.
    assert temporary[9_350] < 15 * temporary[935]
    assert temporary[93_500] < 15 * temporary[9_350]

    # Control: the same inspection does see the all-coordinates Jacobian.
    population, position, weights, offsets = problem(935, np.arange(926, 935))

    def every_coordinate(x, w, o):
        return jax.jacfwd(lambda y: effect_scales(y, population, w, o))(x)[..., population]

    large, full = two_large_axes(
        every_coordinate, position, weights, offsets, dimension=935
    )
    assert "926,935" in large or "935,926" in large
    assert full > 15 * temporary[935]


def test_large_dimension_columns_are_directional_derivatives():
    dimension = 100_000
    population, position, weights, offsets = problem(dimension, [dimension - 1, 0, 27, 891])

    def function(x):
        return effect_scales(x, population, weights, offsets)

    jacobian = population_jacobian(function, position, population)
    assert jacobian.shape == (dimension - 4, 4)
    for column, index in enumerate(population):
        tangent = jnp.zeros(dimension).at[index].set(1.0)
        _, expected = jax.jvp(function, (position,), (tangent,))
        np.testing.assert_allclose(jacobian[:, column], expected, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("population", [[-1], [5], [0.5], [[0]], [1, 1]])
def test_invalid_indices(population):
    with pytest.raises(ValueError):
        population_jacobian(jnp.sin, jnp.zeros(5), population)
