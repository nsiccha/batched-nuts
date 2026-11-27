"""Selected covariance entries from SciPy's limited-memory BFGS operator.

The initializer needs a variance per coordinate and a small population block,
not a dense inverse. Reuse SciPy's two-loop product for each correction prefix
and accumulate only those entries of the inverse-BFGS rank-two update.
"""

import numpy as np
from scipy.optimize import LbfgsInvHessProduct


def lbfgs_covariance_parts(inverse, population):
    """Return ``diag(H)`` and ``H[population, population]`` without forming H.

    ``inverse`` is the ``LbfgsInvHessProduct`` returned by SciPy's L-BFGS-B.
    Its initial inverse is the identity (no added scaling or regularization).
    With d coordinates, m stored corrections and p population coordinates,
    cost is O(d*m**2 + p**2*m), with O(d*m + p**2) total storage including
    the operator's history and finite-value checks. Thus d scaling is linear for fixed
    memory and a bounded population block. No target derivatives are taken.

    For each correction (s, y), v = H_previous @ y and rho = 1/(y @ s):
      H_next = H_previous + (rho + rho**2 * (y @ v)) * s*s.T
               - rho * (s*v.T + v*s.T).
    This is the usual inverse-BFGS update expanded algebraically. SciPy
    computes v using exactly the preceding corrections, in its own order.
    The returned block is symmetrized, matching the frozen initializer.
    """
    if not isinstance(inverse, LbfgsInvHessProduct):
        raise TypeError("expected SciPy's LbfgsInvHessProduct")
    population = np.asarray(population)
    if population.ndim != 1 or (population.size and population.dtype.kind not in "iu"):
        raise ValueError("population must be a one-dimensional integer index array")
    population = population.astype(np.intp, copy=False)
    d = inverse.shape[0]
    if np.any(population < 0) or np.any(population >= d):
        raise ValueError("population indices are out of range")
    if not (np.isfinite(inverse.sk).all() and np.isfinite(inverse.yk).all()
            and np.isfinite(inverse.rho).all()):
        raise FloatingPointError("non-finite L-BFGS correction history")

    diagonal = np.ones(d)
    # Equality rather than eye also preserves repeated selected indices.
    block = (population[:, None] == population[None, :]).astype(np.float64)
    for i, (s, y, rho) in enumerate(zip(inverse.sk, inverse.yk, inverse.rho)):
        prefix = LbfgsInvHessProduct(inverse.sk[:i], inverse.yk[:i])
        v = prefix.matvec(y)
        coefficient = rho * (1.0 + rho * np.dot(y, v))
        diagonal += coefficient * s * s - 2.0 * rho * s * v
        sp, vp = s[population], v[population]
        block += coefficient * np.outer(sp, sp)
        cross = np.outer(sp, vp)
        block -= rho * (cross + cross.T)

    block = 0.5 * (block + block.T)
    if not (np.isfinite(diagonal).all() and np.isfinite(block).all()):
        raise FloatingPointError("non-finite selected L-BFGS covariance")
    return diagonal, block
