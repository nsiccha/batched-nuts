"""Check frozen-source fit equivalence and that the amended steps scale linearly.

Run on the original development targets and synthetic inputs only; never builds
a wren target. Arguments: original moments directory, new amendment directory,
JSON output; --version selects the frozen version (default 1) and --only a
subset of the parts, so that one run fits a short compute allowance.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import sys
import time
import tracemalloc
from pathlib import Path

# Before NumPy/SciPy/JAX imports, including in subprocess-free runs.
os.environ.update(
    OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", JAX_PLATFORMS="cpu"
)
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from amend_moment_initializer import COVARIANCE, FROZEN, GEOMETRY
from lbfgs_covariance import lbfgs_covariance_parts
from scipy.optimize import LbfgsInvHessProduct, minimize

jax.config.update("jax_enable_x64", True)

DEVELOPMENT = ("eight_schools-eight_schools_centered", "seeds_data-seeds_stanified_model")
GEOMETRY_PARTS = ("geometry", "effect_scales")
PARTS = ("scaling", *DEVELOPMENT, *GEOMETRY_PARTS)
CHARTS = ("noncentered", "centered", "moment")
FIELDS = (
    "starts",
    "origin",
    "variance",
    "chart",
    "mean",
    "fitted_variance",
    "fitted_population_cov",
    "fit_chart",
)
DIAGNOSTICS = (
    "updates",
    "joint_gradient_calls",
    "recharts",
    "converged",
    "hit_cap",
    "locate",
    "success",
    "tempering",
    "rejected_updates",
)


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
        methods = {
            "compact": lambda inverse=inverse, pop=pop: lbfgs_covariance_parts(inverse, pop)
        }
        if d == 935:

            def dense(inverse=inverse, pop=pop):
                matrix = inverse.todense()
                matrix = 0.5 * (matrix + matrix.T)
                return np.diag(matrix).copy(), matrix[np.ix_(pop, pop)]

            methods["dense"] = dense
            a, b = methods["compact"](), dense()
            for x, y in zip(a, b, strict=True):
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
            row = dict(
                d=d,
                corrections=10,
                population=4,
                method=method,
                median_seconds=float(np.median(elapsed)),
                peak_traced_bytes=max(peaks),
                correction_bytes=inverse.sk.nbytes + inverse.yk.nbytes,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    compact = [r for r in rows if r["method"] == "compact"]
    # Two tenfold dimension increases must stay far below quadratic growth.
    for a, b in itertools.pairwise(compact):
        assert b["peak_traced_bytes"] < 15 * a["peak_traced_bytes"]
    return rows


def effect_scale_target(effects, seed=11):
    """A synthetic hierarchical normal model with one prior log scale per effect.

    Coordinates are a location and two scale parameters (the population),
    then the noncentered effects. Each effect's log scale depends
    nonlinearly on both scale parameters through its own weights, as a
    spectral or distance-weighted prior scale does, so the scale Jacobian has
    a row per effect.
    """
    rng = np.random.default_rng(seed)
    d = effects + 3
    population, groups = np.arange(3), np.arange(3, d)
    weight = rng.uniform(-0.5, 0.5, (effects, 2))
    noise = 0.5
    y = jnp.asarray(
        1.0
        + np.exp(-0.3 + np.tanh(weight @ np.array([-0.3, 0.4]))) * rng.normal(size=effects)
        + noise * rng.normal(size=effects)
    )
    weight = jnp.asarray(weight)

    def prior(x):
        return x[0] + jnp.zeros(effects), x[1] + jnp.tanh(weight @ x[1:3])

    def to_model(q, c):
        mu, log_scale = prior(q)
        delta = -c * log_scale
        return q.at[groups].set((q[groups] - c * mu) * jnp.exp(delta)), jnp.sum(delta)

    def logp(x):
        mu, log_scale = prior(x)
        z = x[groups]
        return (
            -0.5 * jnp.sum(((y - mu - z * jnp.exp(log_scale)) / noise) ** 2)
            - 0.5 * jnp.sum(z**2)
            - 0.5 * (x[0] / 5.0) ** 2
            - 0.5 * x[1] ** 2
            - 0.5 * x[2] ** 2
        )

    return dict(
        d=d,
        population=population,
        groups=groups,
        chart_identity=np.zeros(effects),
        prior=prior,
        to_model=to_model,
        logp=logp,
    )


def geometry(old, new):
    """Compiled array shapes and values of the low-scale centering update."""
    rows = []
    for d, modules in (
        (935, (("original", old), ("amended", new))),
        (1870, (("original", old), ("amended", new))),
        (9350, (("amended", new),)),
        (93500, (("amended", new),)),
    ):
        target = effect_scale_target(d - 3)
        for method, module in modules:
            engine = module.MomentInitializer(target)
            mean, variance, cov, _ = engine.initial_state(0)
            compiled = engine.information.lower(mean, variance, cov, engine.zero).compile()
            shapes = set(
                re.findall(r"(?:f64|f32|s32|s64|pred)\[([0-9,]+)\]", compiled.as_text())
            )
            large = sorted(
                shape
                for shape in shapes
                if sum(int(n) >= d // 2 for n in shape.split(",")) >= 2
            )
            row = dict(
                d=d,
                population=3,
                method=method,
                two_large_axes=large,
                temporary_bytes=int(compiled.memory_analysis().temp_size_in_bytes),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    by = {(r["d"], r["method"]): r for r in rows}
    # The inspection must see the reported defect before its absence means anything.
    assert by[935, "original"]["two_large_axes"] and by[1870, "original"]["two_large_axes"]
    original = [by[d, "original"]["temporary_bytes"] for d in (935, 1870)]
    assert original[1] > 3 * original[0]
    assert all(not r["two_large_axes"] for r in rows if r["method"] == "amended")
    amended = [by[d, "amended"]["temporary_bytes"] for d in (935, 9350, 93500)]
    for a, b in itertools.pairwise(amended):
        assert b < 15 * a

    # Same values where both can be evaluated.
    target = effect_scale_target(932)
    engines = [module.MomentInitializer(target) for module in (old, new)]
    rng = np.random.default_rng(27)
    parity = []
    for case in range(6):
        factor = rng.normal(size=(3, 3))
        arguments = tuple(
            map(
                jnp.asarray,
                (
                    rng.normal(scale=0.7, size=935),
                    np.exp(rng.normal(scale=0.7, size=935)),
                    0.2 * (factor @ factor.T / 3 + 0.1 * np.eye(3)),
                    rng.uniform(size=932),
                ),
            )
        )
        a, b = (np.asarray(engine.information(*arguments)) for engine in engines)
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-14)
        row = dict(
            case=case,
            identical=bool(np.array_equal(a, b)),
            max_absolute_difference=float(np.max(np.abs(a - b))),
            centering_range=[float(a.min()), float(a.max())],
        )
        parity.append(row)
        print(json.dumps(row), flush=True)
    return dict(compiled=rows, parity=parity)


def compare_fits(engines, label, charts=CHARTS, seed=73):
    rows = []
    for chart in charts:
        a, b = [engine.fit(seed, chart=chart, draws=128) for engine in engines]
        errors, scaled_errors = {}, {}
        for field in FIELDS:
            errors[field] = float(np.max(np.abs(a[field] - b[field])))
            # Full-fit roundoff is amplified by re-charting and small eigensolves.
            # Report it explicitly; require <1 ppm with a unit absolute floor.
            scaled_errors[field] = float(
                np.max(np.abs(a[field] - b[field]) / np.maximum(1.0, np.abs(a[field])))
            )
            assert scaled_errors[field] < 1e-6, (label, chart, field, scaled_errors[field])
        for field in DIAGNOSTICS:
            x, y = a["diagnostics"].get(field), b["diagnostics"].get(field)
            if field == "tempering" and x is not None:
                # Its start temperature is a float: a located start passes its rounding on.
                assert abs(x["beta0"] - y["beta0"]) <= 1e-6 * abs(x["beta0"]), field
                x, y = (dict(each, beta0=None) for each in (x, y))
            assert x == y, field
        row = dict(
            target=label,
            chart=chart,
            seed=seed,
            max_absolute_error=errors,
            max_unit_scaled_error=scaled_errors,
            identical=all(np.array_equal(a[field], b[field]) for field in FIELDS)
            and a["diagnostics"]["history"] == b["diagnostics"]["history"],
            updates=b["diagnostics"]["updates"],
            recharts=b["diagnostics"]["recharts"],
            gradient_calls=b["diagnostics"]["joint_gradient_calls"],
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def prepared(old, new, target, **options):
    engines = [module.MomentInitializer(target, **options) for module in (old, new)]
    for engine in engines:
        engine.prepare()
    return engines


def initial_covariance(engine, target):
    # Isolate the changed operation before stochastic updates can amplify rounding.
    def objective(q):
        value, grad = engine.negative(q)
        return float(value), np.asarray(grad, dtype=np.float64)

    located = minimize(objective, np.zeros(target["d"]), jac=True, method="L-BFGS-B")
    dense = located.hess_inv.todense()
    dense = 0.5 * (dense + dense.T)
    initial = lbfgs_covariance_parts(located.hess_inv, target["population"])
    expected = (np.diag(dense), dense[np.ix_(target["population"], target["population"])])
    covariance_error = []
    for actual, reference in zip(initial, expected, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-12)
        covariance_error.append(float(np.max(np.abs(actual - reference))))
    return covariance_error


def located_fits(engines, target, label, charts):
    covariance_error = initial_covariance(engines[0], target)
    return [
        dict(row, initial_covariance_max_absolute_error=covariance_error)
        for row in compare_fits(engines, label, charts)
    ]


def fit_equivalence(old, new, version, parts):
    rows = []
    for name in DEVELOPMENT:
        if name not in parts:
            continue
        print(f"Preparing differential check: {name}", flush=True)
        target = old.targets.build(name)
        # Defaults are exactly the frozen configuration of each version's harness;
        # the assertions in compare_fits cover both complete fits.
        engines = prepared(old, new, target)
        if COVARIANCE in FROZEN[version]["amendments"]:
            # Version 1: the mode search starts every frozen fit.
            rows += located_fits(engines, target, name, CHARTS)
        else:
            rows += compare_fits(engines, name)
            # Later versions keep the mode search as an option; its start is amended too.
            rows += located_fits(
                prepared(old, new, target, locate=True),
                target,
                f"{name} (locate=True)",
                ("moment",),
            )
    if "effect_scales" in parts:
        # A complete fit in which every effect has its own scale Jacobian row.
        name = "synthetic effect scales (64 effects)"
        print(f"Preparing differential check: {name}", flush=True)
        rows += compare_fits(prepared(old, new, effect_scale_target(64)), name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("amended", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", type=int, choices=sorted(FROZEN), default=1)
    parser.add_argument("--only", nargs="+", choices=PARTS, help="run these parts only")
    args = parser.parse_args()
    source, amended = args.source.resolve(), args.amended.resolve()
    frozen = FROZEN[args.version]
    parts = args.only or [
        part
        for part in PARTS
        if GEOMETRY in frozen["amendments"] or part not in GEOMETRY_PARTS
    ]
    if GEOMETRY not in frozen["amendments"] and set(parts) & set(GEOMETRY_PARTS):
        parser.error(f"version {args.version} has no geometry amendment")
    original_sha256 = hashlib.sha256((source / frozen["file"]).read_bytes()).hexdigest()
    assert original_sha256 == frozen["sha256"]
    manifest = json.loads((amended / frozen["manifest"]).read_text())
    helpers = [amendment["helper"] for amendment in frozen["amendments"]]
    for name in (frozen["file"], *helpers):
        assert (
            hashlib.sha256((amended / name).read_bytes()).hexdigest()
            == manifest["sha256"][name]
        )
    for name in helpers:
        # One helper source: the amended set carries this directory's exact files.
        assert (amended / name).read_bytes() == Path(__file__).with_name(name).read_bytes()
    for path in (
        source,
        source.parent / "variational",
        source / "vendor/vi-ordering",
        source / "vendor/gsmvi",
        amended,
    ):
        sys.path.insert(0, str(path))
    old = load(source / frozen["file"], "frozen_original")
    new = load(amended / frozen["file"], "frozen_amended")
    report = dict(
        numpy=np.__version__,
        scipy=scipy.__version__,
        jax=jax.__version__,
        version=args.version,
        original_sha256=original_sha256,
        amended_sha256=manifest["sha256"],
        parts=parts,
    )
    if "scaling" in parts:
        report["scaling"] = scaling()
    if "geometry" in parts:
        report["geometry"] = geometry(old, new)
    report["fits"] = fit_equivalence(old, new, args.version, parts)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {', '.join(parts)}", flush=True)


if __name__ == "__main__":
    main()
