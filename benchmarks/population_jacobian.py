"""First derivatives with respect to the population coordinates only.

The moment initializer's low-scale centering needs how each effect's prior
log scale changes with the population coordinates. Differentiating with
respect to every coordinate and then keeping the population columns pushes
one tangent per coordinate, so its Jacobian has two axes that grow with the
dimension whenever each effect has its own scale. Here only the population
coordinates carry tangents.
"""

import jax
import numpy as np


def population_jacobian(function, position, population):
    """Return the derivative of ``function(position)`` by ``position[population]``.

    The result equals ``jax.jacfwd(function)(position)[..., population]``: the
    same forward-mode partial derivatives, in the same column order. With d
    coordinates, p population coordinates and n outputs it pushes p tangents
    instead of d, so the work is p forward passes of ``function`` and the
    Jacobian has n*p entries. No array has two axes that grow with d unless
    ``function`` builds one itself. Only first derivatives are taken.

    ``population`` is a one-dimensional array of distinct integer indices
    into ``position``, fixed when the function is traced.
    """
    population = np.asarray(population)
    if population.ndim != 1 or (population.size and population.dtype.kind not in "iu"):
        raise ValueError("population must be a one-dimensional integer index array")
    population = population.astype(np.intp, copy=False)
    dimension = position.shape[0]
    if np.any(population < 0) or np.any(population >= dimension):
        raise ValueError("population indices are out of range")
    if np.unique(population).size != population.size:
        raise ValueError("population indices must be distinct")

    def of_population(values):
        return function(position.at[population].set(values))

    return jax.jacfwd(of_population)(position[population])
