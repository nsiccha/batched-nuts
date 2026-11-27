"""Check frozen-source fit equivalence and selected-covariance memory scaling.

Run on the original development targets only; never builds a wren target.
Arguments: original moments directory, new amendment directory, JSON output.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc

# Before NumPy/SciPy/JAX imports, including in subprocess-free runs.
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  JAX_PLATFORMS="cpu")
import numpy as np
import scipy
from scipy.optimize import LbfgsInvHessProduct, minimize

from amend_moment_initializer import ORIGINAL_SHA256
from lbfgs_covariance import lbfgs_covariance_parts


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scaling():
    rows = []
    for d in (935, 9350, 93500):
        rng = np.random.default_rng(19)
        s = rng.normal(size=(10, d))
        inverse = LbfgsInvHessProduct(s, s * np.geomspace(0.2, 5, d))
        pop = np.array([d - 1, 0, d // 2, d // 3])
        methods = {"compact": lambda: lbfgs_covariance_parts(inverse, pop)}
        if d == 935:
            def dense():
                matrix = inverse.todense()
                matrix = 0.5 * (matrix + matrix.T)
                return np.diag(matrix).copy(), matrix[np.ix_(pop, pop)]
            methods["dense"] = dense
            a, b = methods["compact"](), dense()
            for x, y in zip(a, b):
                np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-13)
        for method, run in methods.items():
            run()  # Warm allocations/imports outside measurements.
            elapsed, peaks = [], []
            for _ in range(5):
                tracemalloc.start()
                before = time.perf_counter()
                diagonal, block = run()
                elapsed.append(time.perf_counter() - before)
                peaks.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
            row = dict(d=d, corrections=10, population=4, method=method,
                       median_seconds=float(np.median(elapsed)),
                       peak_traced_bytes=max(peaks),
                       correction_bytes=inverse.sk.nbytes + inverse.yk.nbytes)
            rows.append(row)
            print(json.dumps(row), flush=True)
    compact = [r for r in rows if r["method"] == "compact"]
    # Two tenfold dimension increases must stay far below quadratic growth.
    for a, b in zip(compact, compact[1:]):
        assert b["peak_traced_bytes"] < 15 * a["peak_traced_bytes"]
    return rows


def fit_equivalence(source, amended):
    for path in (source, source.parent / "variational", source / "vendor/vi-ordering",
                 source / "vendor/gsmvi", amended):
        sys.path.insert(0, str(path))
    old = load(source / "moment_initializer.py", "frozen_original")
    new = load(amended / "moment_initializer.py", "frozen_amended")
    rows = []
    fields = ("starts", "origin", "variance", "chart", "mean", "fitted_variance",
              "fitted_population_cov", "fit_chart")
    for name in ("eight_schools-eight_schools_centered", "seeds_data-seeds_stanified_model"):
        print(f"Preparing differential check: {name}", flush=True)
        target = old.targets.build(name)
        # Defaults are exactly FROZEN in e2e.py; assertions below compare both complete fits.
        engines = [module.MomentInitializer(target) for module in (old, new)]
        for engine in engines:
            engine.prepare()
        # Isolate the changed operation before stochastic updates can amplify rounding.
        def objective(q):
            value, grad = engines[0].negative(q)
            return float(value), np.asarray(grad, dtype=np.float64)
        located = minimize(objective, np.zeros(target["d"]), jac=True, method="L-BFGS-B")
        dense = located.hess_inv.todense()
        dense = 0.5 * (dense + dense.T)
        initial = lbfgs_covariance_parts(located.hess_inv, target["population"])
        expected = (np.diag(dense), dense[np.ix_(target["population"], target["population"])])
        covariance_error = []
        for actual, reference in zip(initial, expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-12)
            covariance_error.append(float(np.max(np.abs(actual - reference))))
        for chart in ("noncentered", "centered", "moment"):
            a, b = [engine.fit(73, chart=chart, draws=128) for engine in engines]
            errors, scaled_errors = {}, {}
            for field in fields:
                errors[field] = float(np.max(np.abs(a[field] - b[field])))
                # Full-fit roundoff is amplified by re-charting and small eigensolves.
                # Report it explicitly; require <1 ppm with a unit absolute floor.
                scaled_errors[field] = float(np.max(
                    np.abs(a[field] - b[field]) / np.maximum(1.0, np.abs(a[field]))))
                assert scaled_errors[field] < 1e-6, (name, chart, field, scaled_errors[field])
            for field in ("updates", "joint_gradient_calls", "recharts", "converged", "hit_cap",
                          "locate", "success"):
                assert a["diagnostics"][field] == b["diagnostics"][field], field
            row = dict(target=name, chart=chart, seed=73, max_absolute_error=errors,
                       max_unit_scaled_error=scaled_errors,
                       initial_covariance_max_absolute_error=covariance_error,
                       updates=b["diagnostics"]["updates"],
                       gradient_calls=b["diagnostics"]["joint_gradient_calls"])
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("amended", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, amended = args.source.resolve(), args.amended.resolve()
    assert hashlib.sha256((source / "moment_initializer.py").read_bytes()).hexdigest() == ORIGINAL_SHA256
    manifest = json.loads((amended / "frozen_method.json").read_text())
    for name in ("moment_initializer.py", "lbfgs_covariance.py"):
        assert hashlib.sha256((amended / name).read_bytes()).hexdigest() == manifest["sha256"][name]
    report = dict(numpy=np.__version__, scipy=scipy.__version__,
                  original_sha256=ORIGINAL_SHA256, amended_sha256=manifest["sha256"],
                  scaling=scaling(), fits=fit_equivalence(source, amended))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: compact covariance, scaling and complete frozen fits", flush=True)


if __name__ == "__main__":
    main()
