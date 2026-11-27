# Compact covariance for the frozen moment initializer

The 27 November 2026 research initializer starts its Gaussian approximation
from SciPy L-BFGS-B's inverse-curvature approximation. Its frozen source hash
is `fc3609757f647a4a9ecadd9d1aeac99d18c45dd58815f5e0174390dbfdf3c216`.
It previously materialized the full matrix to keep only the variances and
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

## Build and verify the amendment

The research source and its dependencies are external inputs. The command
line interface of `amend_moment_initializer.py` takes the existing moments
directory and a **new** output directory. It verifies every source-manifest
hash, requires the exact frozen source above, and emits three files:

- `moment_initializer.py`, with only the covariance extraction and its import changed;
- `lbfgs_covariance.py`, copied from this repository;
- `frozen_method.json`, retaining all other hashes and scientific settings and
  appending an implementation amendment with both old and new source hashes.

The output is a replacement set for that existing research layout, not a
standalone experiment. Its vendor directories and sibling `variational/`
directory remain the dependencies named by the manifest. Preserve the old
freeze and record the new manifest when consuming the replacement. The
builder never changes the input directory or an existing output directory.

`tests/test_lbfgs_covariance.py` compares selected entries with SciPy's full
matrix on bounded inputs and real optimizer histories, checks empty history
and index handling, and checks a 100,000-dimensional problem with full-matrix
methods disabled. `validate_moment_amendment.py` takes the original directory,
the amendment directory and a JSON output path. It measures allocation at
935, 9,350 and 93,500 dimensions and compares complete fits in all three
charts on the original eight-schools and seeds development targets. It
compares starting covariance tightly, reports full-fit rounding differences,
and requires identical optimization diagnostics, update counts, re-chart
counts and stopping decisions. It neither builds nor tunes a wren target.

The script sets serial BLAS and CPU JAX before importing them. Its memory
measurement is peak traced temporary allocation after correction histories
are allocated; the report lists the history size separately. Timings on a
shared host describe this run only.

References: [SciPy's operator and correction-pair interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LbfgsInvHessProduct.html);
[Byrd, Nocedal and Schnabel (1994), compact quasi-Newton representations](https://users.iems.northwestern.edu/~nocedal/Abstracts/representations.html).
