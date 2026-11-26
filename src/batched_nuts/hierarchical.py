"""Starting positions for a hierarchical model, computed from its structure.

The population-level parameters are fitted on their marginal distribution, and the effects
of each group are then drawn from their conditional distribution.  The joint density is
never maximized over all coordinates together: the effects only ever move to their
conditional mode for given population values.  The regions where a group scale collapses,
which attract an optimizer of the joint density, therefore play no role.
"""

import warnings
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class HierarchicalStart(NamedTuple):
    """The result of `hierarchical_start`."""

    position: np.ndarray  # (num_chains, dim) starting positions
    inverse_mass: np.ndarray  # (num_chains, dim) estimated variance of the starts
    population: np.ndarray  # (num_population,) fitted population-level values
    covariance: np.ndarray  # (num_population, num_population) their covariance
    num_converged: int  # how many of the fits converged
    num_agreeing: int  # how many of those ended at the chosen fit
    num_steps: np.ndarray  # (num_fits,) Newton steps each fit took


def _floor(eigenvalues, damping):
    """Positive curvatures: magnitudes, bounded away from zero, plus the damping."""
    top = jnp.max(jnp.abs(eigenvalues))
    tiny = jnp.finfo(eigenvalues.dtype).tiny
    return jnp.maximum(jnp.abs(eigenvalues), 1e-10 * top + tiny) + damping, top


# The blocks are small and there are many of them.  Their Cholesky factors and solves
# are written out entry by entry, so that one array operation handles an entry of every
# block at once; a library call per block is far slower.


def _factor(block):
    """Entries of the lower Cholesky factor of each block, and whether it exists."""
    width = block.shape[-1]
    lower = [[None] * width for _ in range(width)]
    for i in range(width):
        for j in range(i + 1):
            rest = block[..., i, j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = jnp.sqrt(rest) if i == j else rest / lower[j][j]
    pivots = jnp.stack([lower[i][i] for i in range(width)], -1)
    return lower, pivots, jnp.all(jnp.isfinite(pivots) & (pivots > 0))


def _forward(lower, rhs):
    """Solve L y = rhs for each block."""
    y = []
    for i in range(len(lower)):
        y.append((rhs[..., i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i])
    return y


def _backward(lower, y):
    """Solve L' x = y for each block."""
    width = len(lower)
    x = [None] * width
    for i in reversed(range(width)):
        rest = y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, width))
        x[i] = rest / lower[i][i]
    return jnp.stack(x, -1)


def _solve(lower, rhs):
    return _backward(lower, _forward(lower, rhs))


@lru_cache(maxsize=16)
def _compiled(logdensity, with_data, max_steps, num_sweeps):
    def parts(population, groups, data):
        dim = population.shape[0] + groups.size
        width = groups.shape[1]
        # One probe per effect position.  Groups do not interact once the population
        # values are fixed, so `width` Hessian-vector products give every group's block.
        probes = jnp.zeros((width, dim))
        probes = probes.at[jnp.arange(width)[:, None], groups.T].set(1.0)

        def negative(x):
            return -(logdensity(x, data) if with_data else logdensity(x))

        gradient = jax.grad(negative)

        def blocks(x):
            columns = jax.vmap(lambda v: jax.jvp(gradient, (x,), (v,))[1])(probes)
            block = jnp.moveaxis(columns[:, groups], 0, -1)
            return 0.5 * (block + jnp.swapaxes(block, 1, 2))

        def mode(theta, polish=True):
            """The effects that maximize the joint density at these population values."""
            fixed = jax.lax.stop_gradient(theta)
            x = jnp.zeros(dim, theta.dtype).at[population].set(fixed)
            unit = jnp.eye(width, dtype=theta.dtype)

            def body(state):
                x, value, damping, _, count = state
                block = blocks(x)
                top = jnp.max(jnp.abs(jnp.diagonal(block, axis1=-2, axis2=-1)))
                lower, _, positive = _factor(block + damping * unit)
                step = -_solve(lower, gradient(x)[groups])
                moved = x.at[groups].add(step)
                new = negative(moved)
                better = positive & jnp.isfinite(new) & (new <= value)
                small = jnp.max(jnp.abs(step)) <= 1e-8 * (1 + jnp.max(jnp.abs(x[groups])))
                raised = jnp.maximum(4 * damping, 1e-3 * top)
                damping = jnp.where(better, damping / 4, raised)
                return (
                    jnp.where(better, moved, x),
                    jnp.where(better, new, value),
                    damping,
                    better & small,
                    count + 1,
                )

            start = (x, negative(x), jnp.zeros((), theta.dtype), False, 0)
            x, _, _, found, _ = jax.lax.while_loop(
                lambda s: ~s[3] & (s[4] < max_steps), body, start
            )
            # Two plain Newton steps from the converged effects carry the first and
            # second derivatives with respect to the population values.  A caller that
            # needs the value only skips them.
            x = jax.lax.stop_gradient(x).at[population].set(theta)
            for _ in range(2 if polish else 0):
                step = _solve(_factor(blocks(x))[0], gradient(x)[groups])
                x = x.at[groups].add(-step)
            return x, found

        def objective(theta, polish=True):
            """Minus the marginal log density of the population values, by Laplace."""
            x, found = mode(theta, polish)
            # At a conditional mode every block is positive definite.  A stationary
            # point that is not a mode, or a search that did not finish, gives no value.
            _, pivots, positive = _factor(blocks(x))
            value = negative(x) + jnp.sum(jnp.log(pivots))
            return jnp.where(found & positive, value, jnp.inf)

        return blocks, mode, objective

    def fit(theta, population, groups, data):
        """Damped Newton on the marginal, with its exact gradient and Hessian."""
        objective = parts(population, groups, data)[2]

        def derivatives(theta):
            def inner(t):
                value, grad = jax.value_and_grad(objective)(t)
                return grad, (value, grad)

            hessian, (value, grad) = jax.jacfwd(inner, has_aux=True)(theta)
            return value, grad, 0.5 * (hessian + hessian.T)

        def usable(value, grad, hessian):
            return (
                jnp.isfinite(value)
                & jnp.all(jnp.isfinite(grad))
                & jnp.all(jnp.isfinite(hessian))
            )

        def decrement(grad, hessian):
            w, v = jnp.linalg.eigh(hessian)
            # Curvature in every direction, well clear of rounding error: a direction
            # the marginal does not determine must not pass as a maximum.
            convex = jnp.min(w) > 1e-10 * jnp.max(jnp.abs(w))
            return 0.5 * jnp.sum((v.T @ grad) ** 2 / _floor(w, 0.0)[0]), convex

        def body(state):
            theta, value, grad, hessian, damping, _, count = state
            w, v = jnp.linalg.eigh(hessian)
            w, top = _floor(w, damping)
            moved = theta - v @ ((v.T @ grad) / w)
            new = derivatives(moved)
            better = usable(*new) & (new[0] <= value)
            raised = jnp.maximum(4 * damping, 1e-3 * top)
            damping = jnp.where(better, damping / 4, raised)
            theta, value, grad, hessian = jax.tree.map(
                lambda a, b: jnp.where(better, a, b),
                (moved, *new),
                (theta, value, grad, hessian),
            )
            gain, convex = decrement(grad, hessian)
            return theta, value, grad, hessian, damping, convex & (gain <= 1e-9), count + 1

        value, grad, hessian = derivatives(theta)
        ok = usable(value, grad, hessian)
        start = (theta, value, grad, hessian, jnp.ones((), theta.dtype), ~ok, 0)
        theta, value, grad, hessian, _, _, count = jax.lax.while_loop(
            lambda s: ~s[5] & (s[6] < max_steps), body, start
        )
        gain, convex = decrement(grad, hessian)
        converged = ok & usable(value, grad, hessian) & convex & (gain <= 1e-9)
        return theta, value, hessian, count, converged

    def sweep(key, z, theta, factor, objective):
        """One slice update of every whitened coordinate, on the fitted marginal."""

        def density(z):
            value = -objective(theta + factor @ z, polish=False)
            return jnp.where(jnp.isfinite(value), value, -jnp.inf)

        def update(carry, index):
            z, value, key = carry
            key, level_key, bracket_key, point_key = jax.random.split(key, 4)
            level = value - jax.random.exponential(level_key, (), z.dtype)
            origin = z[index]

            def along(u):
                return density(z.at[index].set(u))

            def expand(bound, direction):
                def body(state):
                    u, _, count = state
                    return u + direction, along(u + direction), count + 1

                start = (bound, along(bound), 0)
                return jax.lax.while_loop(
                    lambda s: (s[1] > level) & (s[2] < max_steps), body, start
                )[0]

            # A bracket one fitted standard deviation wide, placed at random around
            # the current value and stepped out until both ends leave the slice.
            lower = origin - jax.random.uniform(bracket_key, (), z.dtype)
            left, right = expand(lower, -1.0), expand(lower + 1.0, 1.0)

            def shrink(state):
                left, right, _, _, key, count = state
                key, point = jax.random.split(key)
                u = jax.random.uniform(point, (), z.dtype, left, right)
                left, right = (
                    jnp.where(u < origin, u, left),
                    jnp.where(u < origin, right, u),
                )
                return left, right, u, along(u), key, count + 1

            state = shrink((left, right, origin, value, point_key, 0))
            _, _, u, new, _, _ = jax.lax.while_loop(
                lambda s: (s[3] < level) & (s[5] < max_steps), shrink, state
            )
            inside = new >= level
            z = z.at[index].set(jnp.where(inside, u, origin))
            return (z, jnp.where(inside, new, value), key), None

        # A normal draw outside the marginal's support starts from the fit instead.
        value = density(z)
        z = jnp.where(jnp.isfinite(value), z, jnp.zeros_like(z))
        value = jnp.where(jnp.isfinite(value), value, density(jnp.zeros_like(z)))
        for sweep_key in jax.random.split(key, num_sweeps):
            (z, value, _), _ = jax.lax.scan(
                update, (z, value, sweep_key), jnp.arange(z.shape[0])
            )
        return z

    def draw(key, theta, factor, population, groups, data):
        """Population values near the fit, then effects given them."""
        blocks, mode, objective = parts(population, groups, data)
        population_key, sweep_key, effects_key = jax.random.split(key, 3)
        z = jax.random.normal(population_key, theta.shape, theta.dtype)
        if num_sweeps:
            z = sweep(sweep_key, z, theta, factor, objective)
        theta = theta + factor @ z
        x, found = mode(theta)
        lower, _, positive = _factor(blocks(x))
        noise = jax.random.normal(effects_key, groups.shape, theta.dtype)
        # With block = L L', the effects mode + solve(L', noise) have its inverse as
        # covariance.
        effects = _backward(lower, [noise[..., i] for i in range(groups.shape[1])])
        return x.at[groups].add(effects), found & positive

    def spread(theta, covariance, population, groups, data):
        """The variance of the starts per coordinate, to first order around the fit."""
        blocks, mode, _ = parts(population, groups, data)
        jacobian = jax.jacfwd(lambda t: mode(t)[0])(theta)
        lower = _factor(blocks(mode(theta)[0]))[0]
        unit = jnp.eye(groups.shape[1], dtype=theta.dtype)
        conditional = jnp.stack(
            [_solve(lower, unit[i])[..., i] for i in range(groups.shape[1])], -1
        )
        marginal = jnp.einsum("ip,pq,iq->i", jacobian, covariance, jacobian)
        return marginal.at[groups].add(conditional)

    return (
        jax.jit(jax.vmap(fit, in_axes=(0, None, None, None))),
        jax.jit(jax.vmap(draw, in_axes=(0, None, None, None, None, None))),
        jax.jit(spread),
    )


def hierarchical_start(
    logdensity,
    population,
    groups,
    num_chains,
    key,
    *,
    data=None,
    num_fits=8,
    initial=None,
    num_sweeps=1,
    max_steps=100,
):
    """Starting positions and a diagonal inverse mass for a hierarchical model.

    The model has population-level parameters and, for each group, a few effects.  Given
    the population-level parameters the groups must be independent of each other: one
    grouping factor, with any number of effects per group.

    Parameters
    ----------
    logdensity
        The joint log density on unconstrained coordinates: `logdensity(position)`, or
        `logdensity(position, data)` when `data` is given.  It must be four times
        differentiable with JAX.  The computation is compiled once per function object;
        pass changing data through `data`, not through a closure.
    population
        Integer coordinates of the population-level parameters, shape (num_population,).
        Only parameters whose prior does not depend on other unknowns belong here.
    groups
        Integer coordinates of the effects, shape (num_groups, effects_per_group).  Row
        `j` holds the effects of group `j`.  Together with `population` it must name
        every coordinate exactly once.
    num_chains
        How many starting positions to draw.
    key
        A JAX PRNG key.
    data
        Passed to `logdensity` as its second argument.
    num_fits
        The fit is repeated from this many starting values and the results are compared.
    initial
        Starting values of the population-level parameters, shape
        (num_fits, num_population).  By default uniform on (-2, 2).
    num_sweeps
        Slice-sampling updates of each chain's population values on the fitted marginal,
        one per coordinate and sweep.  0 keeps the draw from the normal distribution.
    max_steps
        The most steps a fit, the search for a conditional mode, or one slice update
        may take.

    Returns
    -------
    HierarchicalStart
        `position` and `inverse_mass` have one row per chain and fit `Jobs` directly.

    Raises
    ------
    RuntimeError
        If none of the fits converges, if the effects of a chain have no conditional
        mode, or if a starting position is not finite.  Fits that do not converge are
        left out; `num_converged` says how many did.

    Notes
    -----
    For fixed population values, the effects of all groups are moved to their conditional
    mode by Newton steps on one small block per group.  The blocks come from
    `effects_per_group` Hessian-vector products of the joint density.  Laplace's formula
    then gives the marginal density of the population values, which is maximized by a
    damped Newton method with its exact gradient and Hessian.  Each chain's population
    values start as a draw from the normal distribution with that Hessian's inverse as
    covariance, and are then updated on the marginal itself by slice sampling, one
    coordinate of the whitened values at a time, from a bracket one standard deviation
    wide.  With few groups the marginal of a scale parameter is far from normal, and the
    update removes most of that error.  The chain's effects are drawn from the normal
    distribution at their conditional mode.

    When the effects are conditionally normal, as with a normal likelihood that is linear
    in them, the marginal density and the conditional draws are exact.  Otherwise both
    are Laplace approximations.  The fit costs one directional derivative of the marginal
    per population-level parameter and step, so it is meant for models with few of them.
    `inverse_mass` estimates the variance of the starts per coordinate: the fitted
    covariance carried through the conditional modes to first order, plus the conditional
    variance at the fit.  It is exact when the conditional modes are linear in the
    population values.

    The search for the conditional modes starts from zero effects.  Enable 64-bit floats
    in JAX before calling: the stopping tests assume them.
    """
    population = np.asarray(population)
    groups = np.asarray(groups)
    if population.ndim != 1 or groups.ndim != 2:
        raise ValueError(
            "population must have shape (num_population,) and groups "
            "(num_groups, effects_per_group)"
        )
    dim = population.size + groups.size
    named = np.concatenate([population, groups.ravel()])
    if not np.array_equal(np.sort(named), np.arange(dim)):
        raise ValueError(
            f"population and groups must name each of the {dim} coordinates exactly once"
        )
    initial_key, draw_key = jax.random.split(key)
    if initial is None:
        initial = jax.random.uniform(
            initial_key, (num_fits, population.size), minval=-2.0, maxval=2.0
        )
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "hierarchical_start: 64-bit floats are off in JAX; its stopping tests assume "
            "them and fits may not converge.",
            stacklevel=2,
        )
    initial = jnp.asarray(initial, dtype=jnp.result_type(float))
    if initial.shape != (num_fits, population.size):
        raise ValueError(f"initial must have shape {(num_fits, population.size)}")
    if int(num_sweeps) != num_sweeps or num_sweeps < 0:
        raise ValueError("num_sweeps must be a non-negative integer")
    fit, draw, spread = _compiled(
        logdensity, data is not None, int(max_steps), int(num_sweeps)
    )
    population_index, group_index = jnp.asarray(population), jnp.asarray(groups)
    theta, value, hessian, steps, converged = (
        np.asarray(a) for a in fit(initial, population_index, group_index, data)
    )
    converged = converged & np.isfinite(value)
    if not converged.any():
        raise RuntimeError(
            f"hierarchical_start: none of the {num_fits} fits of the population-level "
            "parameters converged.  Check that `population` holds only parameters whose "
            "prior does not depend on other unknowns, that the groups are independent "
            "given them, that every population-level parameter is determined by the data "
            "or its prior, and that the effects have a conditional mode reachable from "
            "zero."
        )
    best = int(np.argmin(np.where(converged, value, np.inf)))
    covariance = np.linalg.inv(hessian[best])
    covariance = 0.5 * (covariance + covariance.T)
    error = np.sqrt(np.diag(covariance))
    apart = np.max(np.abs(theta - theta[best]) / error, axis=1)
    agreeing = int((converged & (apart <= 0.1)).sum())
    if agreeing < converged.sum():
        warnings.warn(
            f"hierarchical_start: {int(converged.sum()) - agreeing} of "
            f"{int(converged.sum())} converged fits ended elsewhere, up to "
            f"{float(np.max(apart[converged])):.3g} standard errors from the chosen one; "
            "the fit with the highest marginal density is used.",
            stacklevel=2,
        )
    chosen = jnp.asarray(theta[best])
    position, found = (
        np.asarray(a)
        for a in draw(
            jax.random.split(draw_key, num_chains),
            chosen,
            jnp.asarray(np.linalg.cholesky(covariance)),
            population_index,
            group_index,
            data,
        )
    )
    if not found.all():
        raise RuntimeError(
            f"hierarchical_start: the effects of {int((~found).sum())} of {num_chains} "
            "chains have no conditional mode at the population values drawn for them."
        )
    variance = np.asarray(
        spread(chosen, jnp.asarray(covariance), population_index, group_index, data)
    )
    if not (np.all(np.isfinite(position)) and np.all(variance > 0)):
        raise RuntimeError(
            "hierarchical_start: a starting position or its variance is not finite."
        )
    return HierarchicalStart(
        position=position,
        inverse_mass=np.tile(variance, (num_chains, 1)),
        population=theta[best],
        covariance=covariance,
        num_converged=int(converged.sum()),
        num_agreeing=agreeing,
        num_steps=steps,
    )
