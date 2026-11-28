# Amendments to the frozen moment initializers

The research moment initializers of 27 and 28 November 2026 are frozen: each
version's manifest records the SHA-256 of its source and of every file it
depends on. Two steps in those sources built arrays whose size grows with the
square of the dimension. An amendment replaces such a step by a computation of
the same quantity that is linear in the dimension, and changes nothing else:
no model, update rule, setting, seed or stopping rule.

| Version | Frozen source | SHA-256 | Amended steps |
|---|---|---|---|
| 1 | `moment_initializer.py` | `fc3609757f647a4a9ecadd9d1aeac99d18c45dd58815f5e0174390dbfdf3c216` | starting covariance |
| 2 | `moment_initializer_v2.py` | `07626d760f5720564e56697b2b17e77dc2f325557c41c3a3a356522851fb232f` | scale derivatives, starting covariance |
| 3 | `moment_initializer_v3.py` | `ced682b9fbd2e9bb251f6322128e16c5b14e0616f191dbc8732d97782cad86f7` | scale derivatives, starting covariance |

## Starting covariance from the mode search

Version 1 starts its Gaussian approximation from SciPy L-BFGS-B's
inverse-curvature approximation. Versions 2 and 3 keep that mode search only
as the option `locate=True`; their frozen configurations do not run it. The
frozen sources materialized the full matrix to keep only the variances and
the small population covariance block.

`lbfgs_covariance.py` computes those same entries directly. For each stored
correction it asks SciPy's `LbfgsInvHessProduct.matvec` for the preceding
inverse applied to the gradient difference, then accumulates the selected
entries of the inverse-BFGS rank-two update. The optimizer, its identity
initial inverse, correction order and memory length are preserved. No
target Hessian or Hessian-vector product is evaluated.

For dimension d, m stored corrections and p population coordinates, the
cost is O(d m² + p² m) time and O(d m + p²) memory. This is linear in d
for fixed m and a bounded p. A growing dense population block remains
quadratic in p, as in the frozen scientific method. This amendment does
not claim to make that separate case linear.

The operation is algebraically equivalent, not bitwise identical. The
initializer still rejects nonfinite selected entries and nonpositive
variances. Roundoff can propagate through later updates and can affect a
threshold decision on an untested target; no universal bitwise trajectory
equivalence is claimed.

## Scale derivatives for the low-scale centering

Versions 2 and 3 evaluate each effect's centering at a low population scale.
For that they need the derivative of every effect's prior log scale with
respect to the population coordinates, which the delta method turns into a
standard deviation of the log scale. The frozen sources computed
`jax.jacfwd` with respect to all d coordinates and then kept the p population
columns. That pushes one tangent per coordinate: a d × d identity and, when
each of J effects has its own scale, a J × d Jacobian. The development
targets declare one or two shared scales, so their Jacobian has one or two
rows; the full cost appears when every effect has its own scale, as in the
935-coordinate case that prompted this amendment.

`population_jacobian.py` differentiates with respect to the population
coordinates only. It pushes p tangents, so the Jacobian has J × p entries and
the work is p forward passes of the prior. These are the same forward-mode
partial derivatives in the same column order; the tangents of the other
coordinates were never used. The centering formula, `z_low` and the delta
method are unchanged, and only first derivatives are taken.

Since the derivatives are the same, the centering update and every complete
fit with a frozen configuration agreed bit for bit wherever the original and
the amended source were compared. A compiler may still order floating-point
operations differently for another target, so, as above, no universal bitwise
equivalence is claimed.

## Build and verify an amendment

The research source and its dependencies are external inputs. The command
line interface of `amend_moment_initializer.py` takes the existing moments
directory, a **new** output directory and `--version` (default 1). It
verifies every source-manifest hash, requires the exact frozen source of
that version, and emits:

- the initializer, with only the passages above and their imports changed;
- `lbfgs_covariance.py`, and from version 2 on `population_jacobian.py`,
  copied from this repository;
- the version's manifest, retaining all other hashes and scientific settings
  and appending one entry per replaced passage. Each entry names the source
  hash before and after that passage and its helper's hash, so consecutive
  entries chain from the frozen source to the emitted one.

The output is a replacement set for that existing research layout, not a
standalone experiment. Its vendor directories and sibling `variational/`
directory remain the dependencies named by the manifest. Preserve the old
freeze and record the new manifest when consuming the replacement. The
builder never changes the input directory or an existing output directory.

`tests/test_lbfgs_covariance.py` compares selected entries with SciPy's full
matrix on bounded inputs and real optimizer histories, checks empty history
and index handling, and checks a 100,000-dimensional problem with full-matrix
methods disabled. `tests/test_population_jacobian.py` compares the Jacobian
with the population columns of the all-coordinates one, inspects the
optimized program's array shapes and temporary memory at 935, 9,350 and
93,500 dimensions against an all-coordinates control, and checks a
100,000-dimensional problem against single directional derivatives.
`tests/test_amend_moment_initializer.py` covers the builder on a stand-in
source and checks that the replaced line and its replacement compute the
same Jacobian.

`validate_moment_amendment.py` takes the original directory, the amendment
directory, a JSON output path and `--version`; `--only` runs a subset of its
parts, so that one run fits a short compute allowance. It never builds a
wren target. Its parts are:

- `scaling`: allocation of the covariance extraction at 935, 9,350 and
  93,500 dimensions;
- one part per development target (eight schools and seeds): complete fits
  in all three charts with the frozen configuration, original against
  amended. For versions 2 and 3 it adds a `locate=True` fit, which is the
  only place their starting covariance is used. Starting covariances are
  compared tightly, full-fit differences are reported, and update counts,
  re-chart counts, stopping decisions and the other discrete diagnostics
  must be identical;
- `geometry` (versions 2 and 3): the optimized program and temporary memory
  of the centering update, original against amended, on a synthetic model
  in which every effect has its own prior scale, and the values of that
  update on random inputs;
- `effect_scales` (versions 2 and 3): complete fits of that synthetic model
  in all three charts.

The script sets serial BLAS and CPU JAX before importing them. The covariance
memory measurement is peak traced temporary allocation after correction
histories are allocated; the report lists the history size separately. The
centering memory figure is the compiler's own account of the optimized
program's temporary buffers. Timings on a shared host describe this run only.

References: [SciPy's operator and correction-pair interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.html);
[Byrd, Nocedal and Schnabel (1994), compact quasi-Newton representations](https://users.iems.northwestern.edu/~nocedal/Abstracts/representations.html);
[JAX forward-mode Jacobians](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html).
