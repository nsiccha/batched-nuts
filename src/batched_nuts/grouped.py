"""Ordered elimination of normal group effects in crossed and nested models."""

import operator
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import ArpackNoConvergence, svds


class GroupFactor(NamedTuple):
    """Coordinate indices by level and the level of each observation (zero based)."""

    coordinates: ArrayLike
    levels: ArrayLike


class GroupedDensity(NamedTuple):
    """Local terms for two crossed or nested sets of conditionally normal effects.

    population(theta, data) returns the population log prior including Jacobians.
    normal(theta, data) returns one (location, scale) pair per factor, broadcastable
    to its coordinate table. Entries are independent normals given theta.
    observation(theta, first, second, i, data) is one scalar log likelihood; first
    and second are vectors for the levels observed in row i. Set quadratic=True
    if every observation is jointly quadratic in the effects at fixed theta.
    """

    population: Callable
    normal: Callable
    observation: Callable
    quadratic: bool = False


def factor_indices(population, factors):
    population = np.asarray(population)
    tables = tuple(np.asarray(f.coordinates) for f in factors)
    levels = tuple(np.asarray(f.levels) for f in factors)
    if population.ndim != 1 or not population.size:
        raise ValueError("population must be a nonempty vector of coordinate indices")
    if any(t.ndim != 2 or not t.size for t in tables):
        raise ValueError("each factor needs a nonempty (levels, effects) coordinate table")
    indices = np.concatenate((population, *(t.ravel() for t in tables)))
    if not np.issubdtype(indices.dtype, np.integer) or not np.array_equal(
        np.sort(indices), np.arange(indices.size)
    ):
        raise ValueError("population and factors must name each coordinate exactly once")
    for table, idx in zip(tables, levels, strict=True):
        if (
            idx.ndim != 1
            or not len(idx)
            or idx.shape != levels[0].shape
            or not np.issubdtype(idx.dtype, np.integer)
            or np.any((idx < 0) | (idx >= len(table)))
        ):
            raise ValueError("each observation must name one valid level in every factor")
    return tables, levels


class _InvalidPoint(RuntimeError):
    """A population proposal where the conditional approximation is undefined."""


class _Precision:
    """Block-diagonal complete information minus a sparse conditional-score covariance."""

    def __init__(self, blocks, coupling, nested):
        groups, width, _ = blocks.shape
        if nested:
            # Every column belongs to at most one block, including repeated observations.
            for i in range(width):
                for j in range(width):
                    correction = coupling[i::width].multiply(coupling[j::width]).sum(1)
                    blocks[:, i, j] -= np.asarray(correction).ravel()
        self.lower = np.linalg.cholesky(blocks)
        self.logdet = 2 * np.log(self.lower.diagonal(axis1=1, axis2=2)).sum()
        self.bound = 0.0
        self.coupling = None
        if nested or not coupling.nnz:
            return

        eye = np.broadcast_to(np.eye(width), blocks.shape)
        inverse = np.linalg.solve(self.lower, eye)
        preconditioner = sparse.bsr_matrix(
            (inverse, np.arange(groups), np.arange(groups + 1)),
            shape=(groups * width, groups * width),
        )
        k = (preconditioner @ coupling).tocsr()
        self.coupling = k
        # Factor only a bounded number of singular directions. The trace of K K'
        # is the sum of squared stored entries, including combined duplicate edges.
        rank = min(k.shape)
        if rank <= 8:
            if k.shape[0] <= k.shape[1]:
                values, self.vectors = np.linalg.eigh((k @ k.T).toarray())
            else:
                values, right = np.linalg.eigh((k.T @ k).toarray())
                keep = values > np.finfo(float).eps * max(1.0, values[-1])
                values = values[keep]
                self.vectors = (k @ right[:, keep]) / np.sqrt(values)
            self.values = np.maximum(values, 0.0)
            self.remainder = 0.0
        else:
            start = np.random.default_rng(0).normal(size=rank)
            vectors, s, _ = svds(k, k=8, v0=start, tol=1e-9)
            self.values, self.vectors = s * s, vectors
            tail = max(0.0, k.data @ k.data - self.values.sum())
            self.remainder = min(tail, self.values.min())
            b = self.remainder
            if b >= 1:
                raise np.linalg.LinAlgError("the marginal effect curvature is indefinite")
            gap = tail * (-np.log1p(-b) / b - 1) if b > 1e-12 else 0.0
            self.logdet -= tail + 0.5 * gap
            self.bound = 0.25 * gap
        if np.any(self.values >= 1):
            raise np.linalg.LinAlgError("the marginal effect curvature is indefinite")
        self.logdet += np.log1p(-self.values).sum()

    def _power(self, values, exponent):
        if self.coupling is None:
            return values
        v, k = self.vectors, self.coupling
        projection = values @ v
        residual = values - projection @ v.T
        result = residual + (projection * (1 - self.values) ** -exponent) @ v.T
        term = residual
        # Apply the binomial series on the complement, bounded by its spectral radius.
        tail = self.remainder
        for step in range(1, 201):
            if tail <= 1e-10:
                return result
            term = ((k @ (k.T @ term.T)).T) * ((step - 1 + exponent) / step)
            result = result + term
            tail *= self.remainder
            if tail / (1 - self.remainder) <= 1e-10:
                return result
        raise _InvalidPoint("inverse curvature series did not converge")

    def transform(self, values, solve=False):
        """Apply an inverse square root to noise, or the inverse to a gradient."""
        shape = values.shape
        cols = values.reshape(-1, *self.lower.shape[:2]).transpose(1, 2, 0)
        if solve:
            cols = np.linalg.solve(self.lower, cols)
        flat = cols.transpose(2, 0, 1).reshape(-1, shape[-1])
        flat = self._power(flat, 1 if solve else 0.5)
        cols = flat.reshape(-1, *self.lower.shape[:2]).transpose(1, 2, 0)
        cols = np.linalg.solve(self.lower.swapaxes(1, 2), cols)
        return cols.transpose(2, 0, 1).reshape(shape)


@lru_cache(maxsize=12)
def _kernels(model, shapes):
    """Compile local derivatives once per model and pair of block shapes."""
    (nfirst, first_width), (nsecond, width) = shapes
    if not model.quadratic and first_width != 1:
        raise ValueError("nonquadratic integration needs a scalar first factor")

    def parameters(theta, data):
        return tuple(
            tuple(jnp.broadcast_to(x, shape) for x in pair)
            for shape, pair in zip(shapes, model.normal(theta, data), strict=True)
        )

    def loss(a, b, idx, theta, data, ma, sa, mb, sb):
        return -model.observation(theta, ma + sa * a, mb + sb * b, idx, data)

    def rows(func, a, b, theta, data, left, right):
        (ma, sa), (mb, sb) = parameters(theta, data)
        return jax.vmap(func, in_axes=(0, 0, 0, None, None, 0, 0, 0, 0))(
            a[left],
            b[right],
            jnp.arange(left.size),
            theta,
            data,
            ma[left],
            sa[left],
            mb[right],
            sb[right],
        )

    def first_conditional(b, theta, data, left, right):
        def stats(a):
            def local(*args):
                grad = jax.grad(loss, 0)
                return grad(*args), jax.jacfwd(grad, 0)(*args)

            g, h = rows(local, a, b, theta, data, left, right)
            return (
                a + jax.ops.segment_sum(g, left, nfirst),
                jnp.eye(first_width) + jax.ops.segment_sum(h, left, nfirst),
            )

        a = jnp.zeros((nfirst, first_width), theta.dtype)
        if model.quadratic:
            g, h = stats(a)
            return -jnp.linalg.solve(h, g[..., None])[..., 0], h

        # For a log-concave likelihood the standard-normal prior gives a
        # curvature lower bound of one. Thus the root of the score lies between
        # zero and minus the score at zero. Safeguard Newton inside this bracket.
        g, h = stats(a)
        lo, hi = jnp.minimum(-g, a), jnp.maximum(-g, a)

        def unsettled(state):
            i, _, _, _, g, h = state
            return (i < 48) & jnp.any(jnp.abs(g) > 1e-8 * jnp.sqrt(h[..., 0]))

        def move(state):
            i, a, lo, hi, g, h = state
            lo = jnp.where(g < 0, a, lo)
            hi = jnp.where(g > 0, a, hi)
            candidate = a - g / h[..., 0]
            inside = (lo <= candidate) & (candidate <= hi)
            a = jnp.where(inside, candidate, (lo + hi) / 2)
            g, h = stats(a)
            return i + 1, a, lo, hi, g, h

        _, a, _, _, g, h = jax.lax.while_loop(unsettled, move, (0, a, lo, hi, g, h))
        h = jnp.where(jnp.abs(g[..., None]) <= 1e-5 * jnp.sqrt(h), h, jnp.nan)
        return jax.lax.stop_gradient(a), jax.lax.stop_gradient(h)

    def grid(b, theta, data, left, right, size=65):
        loc, h = first_conditional(b, theta, data, left, right)
        sd = jax.lax.rsqrt(h[:, 0, 0])
        axis = jnp.linspace(-10.0, 10.0, size, dtype=theta.dtype)

        # Trapezoidal integration, including the Gaussian prior and grid Jacobian.
        def column(point):
            local = rows(loss, point[:, None], b, theta, data, left, right)
            return -0.5 * point**2 - jax.ops.segment_sum(local, left, nfirst)

        def evaluate(radius):
            points = loc + (radius * sd)[:, None] * axis
            heights = jax.vmap(column, in_axes=1, out_axes=1)(points)
            return points, heights

        def truncated(heights):
            peak = jnp.max(heights, axis=1)
            return jnp.maximum(heights[:, 0], heights[:, -1]) > peak - 16.0

        radius = jnp.ones_like(sd)
        points, heights = evaluate(radius)
        tails = truncated(heights)
        radius = jnp.where(tails, 2.0, 1.0)
        points, heights = jax.lax.cond(
            jnp.any(tails), lambda: evaluate(radius), lambda: (points, heights)
        )
        if size > 65:
            # A sampled second effect may make one side of a conditional much
            # broader than at its mode. Extend the finer reconstruction grid.
            tails = truncated(heights)
            radius = jnp.where(tails, 4.0, radius)
            points, heights = jax.lax.cond(
                jnp.any(tails), lambda: evaluate(radius), lambda: (points, heights)
            )
        logw = heights.at[:, 0].add(-jnp.log(2.0)).at[:, -1].add(-jnp.log(2.0))
        total = jax.scipy.special.logsumexp(logw, axis=1)
        value = 0.5 * jnp.sum(b * b) - jnp.sum(
            total + jnp.log(radius * sd * (axis[1] - axis[0]))
        )
        # Compare the integral on the full grid and at every other point.
        coarse = jax.scipy.special.logsumexp(logw[:, ::2], axis=1) + jnp.log(2.0)
        good = (~truncated(heights)) & (jnp.abs(coarse - total) < 1e-4)
        return (
            jnp.where(jnp.all(good), value, jnp.nan),
            points,
            heights,
            jnp.exp(logw - total[:, None]),
        )

    def integrated(flat, theta, data, left, right):
        return grid(flat.reshape(nsecond, width), theta, data, left, right)[0]

    def info(flat, theta, data, left, right):
        b = flat.reshape(nsecond, width)
        eye = jnp.eye(width)
        gb = jax.grad(loss, 1)
        hb = jax.jacfwd(gb, 1)
        if model.quadratic:
            a, h = first_conditional(b, theta, data, left, right)

            def local(*args):
                return loss(*args), gb(*args), hb(*args), jax.jacfwd(gb, 0)(*args)

            val, g, hbb, hab = rows(local, a, b, theta, data, left, right)
            lower = jnp.linalg.cholesky(h)
            norm = jnp.linalg.solve(lower[left], hab.swapaxes(1, 2)).swapaxes(1, 2)
            value = 0.5 * (jnp.sum(a * a) + jnp.sum(b * b)) + val.sum()
            value += jnp.log(jnp.diagonal(lower, axis1=1, axis2=2)).sum()
        else:
            value, points, _, prob = grid(b, theta, data, left, right)

            def column(z):
                def local(*args):
                    return gb(*args), hb(*args)

                return rows(local, z[:, None], b, theta, data, left, right)

            scores, curvatures = jax.vmap(column, in_axes=1, out_axes=-1)(points)
            weights = prob[left]
            g = jnp.einsum("ewq,eq->ew", scores, weights)
            hbb = jnp.einsum("euvq,eq->euv", curvatures, weights)
            norm = (scores - g[..., None]) * jnp.sqrt(weights[:, None, :])
        grad = b + jax.ops.segment_sum(g, right, nsecond)
        blocks = eye + jax.ops.segment_sum(hbb, right, nsecond)
        return value, grad.ravel(), blocks, norm

    def first_draw(flat, theta, data, left, right, variates):
        b = flat.reshape(nsecond, width)
        if model.quadratic:
            a, h = first_conditional(b, theta, data, left, right)
            lower = jnp.linalg.cholesky(h)
            a += jnp.linalg.solve(lower.swapaxes(1, 2), variates[..., None])[..., 0]
        else:
            value, points, heights, _ = grid(b, theta, data, left, right, size=257)
            heights = jnp.where(jnp.isfinite(value), heights, jnp.nan)
            y = jnp.exp(heights - heights.max(1, keepdims=True))
            cdf = jnp.cumsum((y[:, :-1] + y[:, 1:]) / 2, 1)
            targets = variates[:, 0] * cdf[:, -1]
            i = jnp.minimum((cdf < targets[:, None]).sum(1), points.shape[1] - 2)
            j = jnp.arange(nfirst)
            f = targets - jnp.where(i > 0, cdf[j, jnp.maximum(i - 1, 0)], 0.0)
            y0, y1 = y[j, i], y[j, i + 1]
            den = y0 + jnp.sqrt(jnp.maximum(0.0, y0 * y0 + 2 * (y1 - y0) * f))
            a = (
                points[j, i]
                + (points[j, i + 1] - points[j, i])
                * (2 * f / jnp.maximum(den, jnp.finfo(y.dtype).tiny))
            )[:, None]
        (ma, sa), (mb, sb) = parameters(theta, data)
        valid = jnp.all(sa > 0) & jnp.all(sb > 0)
        return jnp.where(valid, ma + sa * a, jnp.nan), jnp.where(
            valid, mb + sb * b, jnp.nan
        )

    def third(flat, theta, data, left, right, directions):
        grad = lambda x: jax.grad(integrated)(x, theta, data, left, right)

        def apply(z):
            return jax.jvp(lambda x: jax.jvp(grad, (x,), (z,))[1], (flat,), (z,))[1]

        return jax.lax.map(apply, directions).mean(0)

    def prior(theta, data):
        params = parameters(theta, data)
        valid = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(scale) & (scale > 0))
                    for mean, scale in params
                ]
            )
        )
        return jnp.where(valid, model.population(theta, data), -jnp.inf)

    return tuple(
        jax.jit(f)
        for f in (
            jax.value_and_grad(integrated),
            info,
            first_draw,
            third,
            prior,
        )
    )


class _Marginal:
    """Eliminate the first factor by level, then integrate the coupled second factor."""

    def __init__(self, model, tables, levels, data, max_steps):
        self.model, self.tables, self.data = model, tables, data
        self.max_steps = max_steps
        left, right = levels
        self.args = (data, jnp.asarray(left), jnp.asarray(right))
        self.kernels = _kernels(model, tuple(t.shape for t in tables))
        pairs = np.unique(np.stack((left, right), 1), axis=0)
        self.nested = np.unique(pairs[:, 0]).size == len(pairs)
        nfirst, first_width = tables[0].shape
        _, width = tables[1].shape
        nodes = first_width if model.quadratic else 65
        self.row = np.broadcast_to(
            right[:, None, None] * width + np.arange(width)[None, :, None],
            (len(left), width, nodes),
        ).ravel()
        self.col = np.broadcast_to(
            left[:, None, None] * nodes + np.arange(nodes), (len(left), width, nodes)
        ).ravel()
        self.shape = (tables[1].size, nfirst * nodes)

    def point(self, theta):
        evaluate, info, _, _, prior = self.kernels
        lp = float(prior(theta, self.data))
        if not np.isfinite(lp):
            raise _InvalidPoint("invalid population density or normal parameters")
        x = np.zeros(self.shape[0])
        if not self.model.quadratic:

            def objective(v):
                f, g = evaluate(v, theta, *self.args)
                return float(f), np.asarray(g)

            fit = minimize(
                objective,
                x,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": self.max_steps,
                    "ftol": 1e-14,
                    "gtol": 1e-7,
                },
            )
            x = fit.x

        def curvature_at(position):
            value, grad, blocks, norm = (
                np.asarray(v) for v in info(position, theta, *self.args)
            )
            if not all(np.isfinite(v).all() for v in (value, grad, blocks, norm)):
                raise _InvalidPoint(
                    "unresolved conditional integral or nonfinite effect derivatives"
                )
            coupling = sparse.coo_matrix(
                (norm.ravel(), (self.row, self.col)), shape=self.shape
            ).tocsr()
            curvature = _Precision(blocks.copy(), coupling, self.nested)
            step = curvature.transform(grad, solve=True)
            gain = 0.5 * (grad @ step)
            return value, step, gain, curvature

        value, step, gain, curvature = curvature_at(x)
        if not self.model.quadratic:
            if gain > 1e-7:
                raise _InvalidPoint("conditional effect fit did not converge")
            # The determinant depends on the outer effect mode. Polish the mode
            # before taking its determinant so that finite differences of the
            # population objective do not amplify residual optimizer error.
            x = x - step
            value, step, gain, curvature = curvature_at(x)
        mode = x - step
        return float(lp - value + gain - 0.5 * curvature.logdet), mode, curvature

    def logdensity(self, theta):
        try:
            return self.point(theta)[0]
        except (
            _InvalidPoint,
            FloatingPointError,
            np.linalg.LinAlgError,
            ArpackNoConvergence,
        ):
            # An infeasible proposal is outside the numerical marginal; the fit and
            # every draw subsequently require finite, positive curvature at their point.
            return -np.inf

    def draw(self, rng, theta):
        _, mode, precision = self.point(theta)
        x = mode + precision.transform(rng.normal(size=mode.size))
        if not self.model.quadratic:
            probes = rng.choice([-1.0, 1.0], size=(12, mode.size))
            directions = precision.transform(probes)
            third = np.asarray(self.kernels[3](mode, theta, *self.args, directions))
            x -= 0.5 * precision.transform(third, solve=True)
        shape = self.tables[0].shape
        u = rng.normal(size=shape) if self.model.quadratic else rng.random(size=shape)
        first, second = self.kernels[2](x, theta, *self.args, u)
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise _InvalidPoint(
                "conditional draw lies outside the resolved integration grid"
            )
        return first, second, precision.bound


def _hessian(f, x):
    """Central differences of the small population marginal at its fitted mode."""
    unit = np.diag(0.002 * np.maximum(1.0, np.abs(x)))
    ans = np.empty((x.size, x.size))
    f0 = f(x)
    for i, ei in enumerate(unit):
        ans[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / ei[i] ** 2
        for j, ej in enumerate(unit[:i]):
            v = f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ans[i, j] = ans[j, i] = v / (4 * ei[i] * ej[j])
    return ans


def _population_draw(rng, mode, lower, f, sweeps, max_steps):
    z = rng.normal(size=mode.size)
    if not sweeps:
        return mode + lower @ z
    val = f(mode + lower @ z)
    if not np.isfinite(val):
        z = np.zeros_like(z)
        val = f(mode)
    for _ in range(sweeps):
        for i in range(z.size):
            threshold = val - rng.exponential()
            origin = z[i]
            lo = origin - rng.uniform()
            hi = lo + 1

            def along(q, i=i):
                z[i] = q
                return f(mode + lower @ z)

            for _ in range(max_steps):
                if along(lo) <= threshold:
                    break
                lo -= 1
            for _ in range(max_steps):
                if along(hi) <= threshold:
                    break
                hi += 1
            for _ in range(max_steps):
                q = rng.uniform(lo, hi)
                proposal = along(q)
                if proposal >= threshold:
                    val = proposal
                    break
                if q < origin:
                    lo = q
                else:
                    hi = q
            else:
                z[i] = origin
    return mode + lower @ z


def grouped_start(
    model,
    population,
    factors,
    num_chains,
    key,
    data,
    num_fits,
    initial,
    num_sweeps,
    max_steps,
):
    """Fit the small population marginal and reconstruct the factors in reverse order."""
    from batched_nuts.hierarchical import HierarchicalStart

    if len(factors) != 2 or not isinstance(model, GroupedDensity):
        raise ValueError("two ordered GroupFactor objects require a GroupedDensity")
    if not jax.config.jax_enable_x64:
        raise ValueError("enable 64-bit floats in JAX for multi-factor integration")
    num_chains, num_fits, max_steps, num_sweeps = map(
        operator.index, (num_chains, num_fits, max_steps, num_sweeps)
    )
    if min(num_chains, num_fits, max_steps) < 1 or num_sweeps < 0:
        raise ValueError(
            "positive chain, fit and step counts and nonnegative sweeps required"
        )
    pop = np.asarray(population)
    tables, levels = factor_indices(pop, factors)
    marginal = _Marginal(model, tables, levels, data, max_steps)
    rng = np.random.default_rng(np.asarray(jax.random.key_data(key)).ravel())
    if initial is None:
        initial = rng.uniform(-2.0, 2.0, size=(num_fits, pop.size))
        initial[0] = 0.0
    if np.shape(initial) != (num_fits, pop.size):
        raise ValueError(f"initial must have shape {(num_fits, pop.size)}")
    steps = np.zeros(num_fits, dtype=int)
    fits = []

    def objective(h):
        return -marginal.logdensity(h)

    def derivative(h):
        # These derivatives include the determinant of the coupled effect
        # curvature. The step must resolve it above the inner optimizer tolerance.
        delta = 0.001 * np.maximum(1.0, np.abs(h))
        return np.asarray(
            [
                (objective(h + d) - objective(h - d)) / (2 * s)
                for d, s in zip(np.diag(delta), delta, strict=True)
            ]
        )

    for i, start in enumerate(initial):
        if not np.isfinite(objective(start)):
            continue
        result = minimize(
            objective,
            start,
            method="BFGS",
            jac=derivative,
            options={"gtol": 3e-5, "maxiter": max_steps},
        )
        steps[i] = result.nit
        if not np.isfinite(result.fun) or not np.isfinite(result.jac).all():
            continue
        hessian = _hessian(objective, result.x)
        if not np.isfinite(hessian).all():
            continue
        ev = np.linalg.eigvalsh(hessian)
        if ev[0] <= 1e-10 * ev[-1]:
            continue
        cov = np.linalg.inv(hessian)
        # The residual Newton displacement is below 0.001 posterior standard
        # deviations in the fitted population geometry.
        if result.jac @ cov @ result.jac <= 1e-6:
            fits.append((result.fun, result.x, cov))
    if not fits:
        raise RuntimeError(
            "no fit of the integrated multi-factor population density converged"
        )
    _, mode, cov = min(fits, key=lambda f: f[0])
    lower = np.linalg.cholesky(cov)
    agree = sum(
        np.max(np.abs(h - mode) / np.sqrt(cov.diagonal())) <= 0.1 for _, h, _ in fits
    )
    if agree < len(fits):
        warnings.warn("converged population fits ended at different modes", stacklevel=3)
    # Replicated starts estimate total marginal variance, including variability of the
    # conditional mean as population parameters change. Small requests use pilot draws.
    n = max(16, num_chains)
    out = np.empty((n, pop.size + sum(t.size for t in tables)))
    bounds = []
    for row in out:
        h = _population_draw(rng, mode, lower, marginal.logdensity, num_sweeps, max_steps)
        a, b, error = marginal.draw(rng, h)
        row[pop], row[tables[0]], row[tables[1]] = h, a, b
        bounds.append(error)
    variance = out.var(axis=0, ddof=1)
    if not np.isfinite(out).all() or not np.all(np.isfinite(variance) & (variance > 0)):
        raise RuntimeError("nonfinite multi-factor starts or degenerate estimated variance")
    if max(bounds) > 0.01:
        warnings.warn(
            f"crossed log-marginal determinant half-error bound reaches "
            f"{max(bounds):.3g} log units at the drawn populations",
            stacklevel=3,
        )
    return HierarchicalStart(
        out[:num_chains].copy(),
        np.tile(variance, (num_chains, 1)),
        mode,
        cov,
        len(fits),
        int(agree),
        steps,
    )
