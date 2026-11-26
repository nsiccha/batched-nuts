"""Ordered elimination agrees with independent Gaussian and quadrature references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit, logsumexp

from batched_nuts import GroupedDensity, GroupFactor, hierarchical_start
from batched_nuts.grouped import _kernels, _Precision


@pytest.mark.parametrize("nested", [False, True])
def test_ordered_starts_against_joint_gaussian(nested):
    left = np.tile(np.arange(4), 3)
    right = left // 2 if nested else (left + np.repeat([0, 1, 1], 4)) % 2
    covariate = np.linspace(-1, 1, len(left))
    y = np.cos(np.arange(len(left))) + 0.6
    factors = (
        GroupFactor(np.arange(1, 9).reshape(4, 2), left),
        GroupFactor(np.arange(9, 11)[:, None], right),
    )
    data = (jnp.asarray(covariate), jnp.asarray(y))
    model = GroupedDensity(
        lambda h, data: -0.5 * h[0] ** 2,
        lambda h, data: ((0.0, jnp.array([1.0, 0.5])), (0.0, 0.8)),
        lambda h, a, b, i, data: (
            -0.5 * ((data[1][i] - h[0] - a[0] - a[1] * data[0][i] - b[0]) / 0.7) ** 2
        ),
        quadratic=True,
    )
    design = np.zeros((len(left), 11))
    design[:, 0] = 1
    rows = np.arange(len(left))
    design[rows, 1 + 2 * left] = 1
    design[rows, 2 + 2 * left] = covariate
    design[rows, 9 + right] = 1
    sd = np.r_[1, np.tile([1.0, 0.5], 4), 0.8, 0.8]
    cov = np.linalg.inv(np.diag(sd**-2) + design.T @ design / 0.7**2)
    mean = cov @ design.T @ y / 0.7**2
    out = hierarchical_start(
        model, [0], factors, 768, jax.random.key(21), data=data, num_fits=1, num_sweeps=0
    )
    np.testing.assert_allclose(out.population, mean[:1], atol=1e-6)
    np.testing.assert_allclose(out.covariance, cov[:1, :1], atol=1e-6)
    z = np.linalg.solve(np.linalg.cholesky(cov), (out.position - mean).T)
    assert np.abs(z.mean(1)).max() < 0.15
    assert np.abs(np.cov(z) - np.eye(len(mean))).max() < 0.22
    np.testing.assert_allclose(out.inverse_mass[0], out.position.var(0, ddof=1))


def test_sparse_curvature_tail_bound_and_solve():
    rng = np.random.default_rng(1)
    coupling = sparse.random(31, 25, density=0.26, random_state=rng, format="csr")
    coupling *= 0.65 / np.linalg.norm(coupling.toarray(), ord=2)
    precise = np.eye(31) - (coupling @ coupling.T).toarray()
    fit = _Precision(np.ones((31, 1, 1)), coupling, nested=False)
    assert abs(fit.logdet - np.linalg.slogdet(precise)[1]) / 2 <= fit.bound + 1e-12
    v = rng.normal(size=(2, 31))
    np.testing.assert_allclose(
        fit.transform(v, solve=True), np.linalg.solve(precise, v.T).T, atol=1e-8
    )


def test_nonlinear_first_integral_and_gradient_against_gauss_hermite():
    model = GroupedDensity(
        lambda h, d: -(h[0] ** 2) / 2,
        lambda h, d: ((0.0, 1.0), (0.0, 1.0)),
        lambda h, a, b, i, d: -jax.nn.softplus(-d[i] * (h[0] + a[0] + b[0])),
    )
    eval_first = _kernels(model, ((2, 1), (2, 1)))[0]
    left = np.array([0, 0, 1, 1, 1])
    right = np.array([0, 1, 0, 1, 1])
    labels = np.array([1, -1, 1, 1, -1])
    x, w = np.polynomial.hermite.hermgauss(80)

    def exact(b):
        total = b @ b / 2
        for g in range(2):
            rows = left == g
            eta = 0.25 + np.sqrt(2) * x[:, None] + b[right[rows]]
            total -= logsumexp(
                np.log(w) + np.log(expit(labels[rows] * eta)).sum(1)
            ) + 0.5 * np.log(2)
        return total

    b = np.array([0.2, -0.55])
    value, grad = eval_first(
        b, jnp.array([0.25]), jnp.array(labels), jnp.array(left), jnp.array(right)
    )
    np.testing.assert_allclose(value, exact(b), atol=1e-8)
    expected = [(exact(b + e) - exact(b - e)) / 2e-5 for e in np.eye(2) * 1e-5]
    np.testing.assert_allclose(grad, expected, atol=1e-8)


def test_one_factor_alias_preserves_draws_and_diagnostics():
    def logp(x):
        return -0.5 * (jnp.sum(x**2) + jnp.sum((x[1:] + x[0] - 1.0) ** 2))

    table = np.array([[1], [2]])

    def call(groups):
        return hierarchical_start(logp, [0], groups, 8, jax.random.key(2), num_fits=1)

    for old, new in zip(call(table), call([GroupFactor(table, [0, 0, 1])]), strict=True):
        np.testing.assert_array_equal(old, new)


def test_malformed_callbacks_and_memberships_are_visible():
    def malformed(h, d):
        raise ValueError("missing supplied normal scale")

    model = GroupedDensity(
        lambda h, d: -(h[0] ** 2) / 2, malformed, lambda *args: 0.0, True
    )
    a = GroupFactor([[1]], [0])
    b = GroupFactor([[2]], [0])

    def run(factors):
        return hierarchical_start(model, [0], factors, 1, jax.random.key(5), num_fits=1)

    with pytest.raises(ValueError, match="valid level"):
        run([a, GroupFactor([[2]], [1])])
    with pytest.raises(ValueError, match="missing supplied normal scale"):
        run([a, b])


def test_single_chain_nonlinear_ordered_start_has_finite_metric():
    model = GroupedDensity(
        lambda h, d: -(h[0] ** 2) / 2,
        lambda h, d: ((0.0, 1.0), (0.0, 1.0)),
        lambda h, a, b, i, d: -jax.nn.softplus(-d[i] * (h[0] + a[0] + b[0])),
    )
    factors = (GroupFactor([[1], [2]], [0, 0, 1, 1]), GroupFactor([[3], [4]], [0, 1, 0, 1]))
    out = hierarchical_start(
        model,
        [0],
        factors,
        1,
        jax.random.key(15),
        data=jnp.array([1, -1, -1, 1]),
        num_fits=1,
    )
    assert out.position.shape == out.inverse_mass.shape == (1, 5)
    assert np.isfinite(out.position).all()
    assert np.isfinite(out.inverse_mass).all() and (out.inverse_mass > 0).all()
