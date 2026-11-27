"""Compare selected entries with SciPy, including real optimizer histories."""

import numpy as np
import pytest
from scipy.optimize import LbfgsInvHessProduct, minimize

from lbfgs_covariance import lbfgs_covariance_parts


@pytest.mark.parametrize("d,m", [(1, 0), (1, 1), (9, 0), (9, 1), (37, 10), (935, 10)])
@pytest.mark.parametrize("selection", ["none", "one", "scattered", "repeated"])
def test_selected_entries(d, m, selection):
    rng = np.random.default_rng(d + m)
    s = rng.normal(size=(m, d))
    y = s * np.exp(rng.uniform(-3, 3, d)) + 0.01 * rng.normal(size=(m, d))
    inverse = LbfgsInvHessProduct(s, y)
    pop = {"none": [], "one": [d - 1], "scattered": list(range(d - 1, -1, -3)),
           "repeated": [d - 1, 0, d - 1]}[selection]
    original = [x.copy() for x in (inverse.sk, inverse.yk, inverse.rho)]
    dense = inverse.todense()
    dense = 0.5 * (dense + dense.T)
    diagonal, block = lbfgs_covariance_parts(inverse, pop)
    np.testing.assert_allclose(diagonal, np.diag(dense), rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(block, dense[np.ix_(pop, pop)], rtol=2e-12, atol=2e-13)
    for actual, expected in zip((inverse.sk, inverse.yk, inverse.rho), original):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("d,maxcor", [(17, 3), (73, 10), (935, 10)])
def test_optimizer_history(d, maxcor):
    # Nonlinear first-order target; enough iterations to wrap the correction buffer.
    scale = np.geomspace(0.1, 100.0, d)
    target = np.linspace(-1, 1, d)

    def objective(x):
        z = x - target
        return np.sum(0.5 * scale * z**2 + 0.01 * z**4), scale * z + 0.04 * z**3

    result = minimize(objective, np.zeros(d), jac=True, method="L-BFGS-B",
                      options={"maxcor": maxcor})
    assert result.nit > maxcor
    dense = result.hess_inv.todense()
    dense = 0.5 * (dense + dense.T)
    pop = np.array([d - 1, 0, d // 2])
    diagonal, block = lbfgs_covariance_parts(result.hess_inv, pop)
    np.testing.assert_allclose(diagonal, np.diag(dense), rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(block, dense[np.ix_(pop, pop)], rtol=2e-11, atol=2e-12)


def test_large_dimension_never_materializes_inverse(monkeypatch):
    d, m = 100_000, 10
    rng = np.random.default_rng(73)
    s = rng.normal(size=(m, d))
    inverse = LbfgsInvHessProduct(s, s * np.linspace(0.5, 2, d))

    def forbidden(*args, **kwargs):
        raise AssertionError("full inverse materialization is forbidden")

    monkeypatch.setattr(LbfgsInvHessProduct, "todense", forbidden)
    monkeypatch.setattr(LbfgsInvHessProduct, "_matmat", forbidden)
    pop = np.array([d - 1, 0, 27, 891])
    diagonal, block = lbfgs_covariance_parts(inverse, pop)
    assert diagonal.shape == (d,)
    assert block.shape == (4, 4)
    assert np.all(diagonal > 0)
    # Independent selected-column checks use only a fixed number of basis vectors.
    for j, index in enumerate(pop):
        basis = np.zeros(d)
        basis[index] = 1
        column = inverse.matvec(basis)
        np.testing.assert_allclose(diagonal[index], column[index], rtol=2e-12)
        np.testing.assert_allclose(block[:, j], column[pop], rtol=2e-12, atol=2e-14)


def test_invalid_history_is_visible():
    inverse = LbfgsInvHessProduct(np.ones((1, 4)), np.ones((1, 4)))
    inverse.sk[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="history"):
        lbfgs_covariance_parts(inverse, [0])


@pytest.mark.parametrize("population", [[-1], [3], [0.5], [[0]]])
def test_invalid_indices(population):
    inverse = LbfgsInvHessProduct(np.empty((0, 3)), np.empty((0, 3)))
    with pytest.raises(ValueError):
        lbfgs_covariance_parts(inverse, population)
