"""Compact per-chain square roots of the inverse mass matrix."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular


class MetricFactor(NamedTuple):
    """L = diag(outer) H₁…Hₖ blockdiag(lower, I) diag(scale); inverse mass = L Lᵀ.

    Each field has a leading job axis. Each row of reflections is a unit
    Householder vector (a zero row is the identity). The triangular block acts
    on the first r coordinates. This representation preserves the frame as well
    as its covariance, including after a coordinatewise scale is fitted.
    """

    outer: Any  # (jobs, dimension)
    reflections: Any  # (jobs, reflections, dimension)
    lower: Any  # (jobs, rank, rank)
    scale: Any  # (jobs, dimension)

    @classmethod
    def from_low_rank(cls, diagonal, basis, core):
        """Factor diag(diagonal) + basis @ core @ basis.T for each job.

        Shapes are (jobs, dim), (jobs, dim, rank) and (jobs, rank, rank).
        The correction may be indefinite if the resulting covariance is positive
        definite. A thin QR and a small Cholesky preserve the supplied column order.
        """
        diagonal, basis, core = map(np.asarray, (diagonal, basis, core))
        if diagonal.ndim != 2 or basis.ndim != 3 or basis.shape[:2] != diagonal.shape:
            raise ValueError("Expected diagonal (jobs, dim) and basis (jobs, dim, rank).")
        jobs, dim, rank = basis.shape
        if core.shape != (jobs, rank, rank):
            raise ValueError("core must have shape (jobs, rank, rank).")
        if not np.all(np.isfinite(diagonal) & (diagonal > 0)):
            raise ValueError("diagonal must be finite and positive.")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(core)):
            raise ValueError("basis and core must be finite.")
        if not np.allclose(core, core.swapaxes(-1, -2)):
            raise ValueError("core must be symmetric.")
        outer = np.sqrt(diagonal)
        width = min(dim, rank)
        dtype = np.result_type(outer, basis, core)
        vectors = np.zeros((jobs, width, dim), dtype)
        lower = np.empty((jobs, width, width), dtype)
        if width:
            raw, tau = np.linalg.qr(basis / outer[:, :, None], mode="raw")
            r = np.triu(raw.swapaxes(-1, -2)[:, :width])
            lower = np.linalg.cholesky(np.eye(width, dtype=dtype) + r @ core @ r.mT)
            for i in range(width):
                vectors[:, i, i] = 1
                vectors[:, i, i + 1 :] = raw[:, i, i + 1 :]
            vectors *= np.sqrt(tau / 2)[:, :, None]
        return cls(outer, vectors, lower, np.ones_like(outer))


def _validate_factor(metric, shape, dtype):
    metric = MetricFactor(*(np.asarray(x, dtype) for x in metric))
    outer, vectors, block, scale = metric
    jobs, dim = shape
    if outer.shape != shape or scale.shape != shape:
        raise ValueError("Metric factor scales must match jobs.position.")
    if vectors.ndim != 3 or vectors.shape[::2] != (jobs, dim):
        raise ValueError("reflections must have shape (jobs, reflections, dim).")
    if (
        block.ndim != 3
        or block.shape[0] != jobs
        or block.shape[1] != block.shape[2]
        or block.shape[1] > dim
    ):
        raise ValueError("lower must have shape (jobs, rank, rank) with rank <= dim.")
    if not all(np.isfinite(x).all() for x in metric):
        raise ValueError("Metric factor entries must be finite.")
    if (
        np.any(outer <= 0)
        or np.any(scale <= 0)
        or np.any(block.diagonal(axis1=1, axis2=2) <= 0)
    ):
        raise ValueError("Metric scales and the diagonal of lower must be positive.")
    if np.any(np.triu(block, 1) != 0):
        raise ValueError("lower must be lower triangular.")
    norms = np.linalg.norm(vectors, axis=-1)
    if np.any((norms != 0) & ~np.isclose(norms, 1, rtol=1e-5, atol=0)):
        raise ValueError("Reflection rows must be unit vectors or zero padding.")
    # A single-precision factor may be passed to a double-precision job.
    rescale = (norms != 0) & (np.abs(norms - 1) > 8 * np.finfo(dtype).eps)
    if np.any(rescale):
        metric = metric._replace(
            reflections=vectors / np.where(rescale, norms, 1)[..., None]
        )
    return metric


def _reflect(x, v):
    return x - 2 * (x @ v)[..., None] * v


def _apply(metric, x, *, transpose=False, solve=False):
    """Apply L, Lᵀ, L⁻¹ or L⁻ᵀ to vectors along the last axis of x."""
    outer, vectors, block, scale = metric
    reverse = transpose != solve
    first, last = (outer, scale) if reverse else (scale, outer)
    x = x / first if solve else x * first

    def reflect(y):
        if vectors.shape[0]:

            def step(i, value):
                row = i if reverse else vectors.shape[0] - 1 - i
                return _reflect(value, vectors[row])

            return jax.lax.fori_loop(0, vectors.shape[0], step, y)
        return y

    if reverse:
        x = reflect(x)
    rank = block.shape[0]
    if rank:
        head = x[..., :rank]
        if solve:
            head = solve_triangular(
                block, head.T, lower=True, trans="T" if transpose else "N"
            ).T
        else:
            head = head @ (block if transpose else block.T)
        x = jnp.concatenate((head, x[..., rank:]), axis=-1)
    if not reverse:
        x = reflect(x)
    return x / last if solve else x * last


def _velocity(metric, momentum):
    if isinstance(metric, MetricFactor):
        return _apply(metric, _apply(metric, momentum, transpose=True))
    return metric * momentum


def _momentum(metric, noise):
    if isinstance(metric, MetricFactor):
        return _apply(metric, noise, transpose=True, solve=True)
    return (1 / jnp.sqrt(metric)) * noise
