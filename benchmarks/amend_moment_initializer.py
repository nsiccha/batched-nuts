"""Build an immutable amendment of the 27 Nov frozen research initializer.

The research run's directory is an input, never an output. This emits the two
Python files and updated manifest into a new directory for review/distribution;
all other files named by the manifest remain the original frozen dependencies.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ORIGINAL_SHA256 = "fc3609757f647a4a9ecadd9d1aeac99d18c45dd58815f5e0174390dbfdf3c216"
OLD = """            inverse = np.asarray(res.hess_inv.todense())
            inverse = .5 * (inverse + inverse.T)
            start_var = np.diag(inverse).copy()
            if not (np.isfinite(inverse).all() and (start_var > 0).all()):
                raise FloatingPointError("mode search returned an unusable inverse-Hessian approximation")
            state = (jnp.asarray(res.x), jnp.asarray(start_var),
                     jnp.asarray(inverse[np.ix_(self.pop, self.pop)]), state[3])"""
NEW = """            start_var, start_cov = lbfgs_covariance_parts(res.hess_inv, self.pop)
            if not (np.isfinite(start_var).all() and np.isfinite(start_cov).all()
                    and (start_var > 0).all()):
                raise FloatingPointError("mode search returned an unusable inverse-Hessian approximation")
            state = (jnp.asarray(res.x), jnp.asarray(start_var),
                     jnp.asarray(start_cov), state[3])"""


def build_amendment(source, output):
    source, output = Path(source), Path(output)
    original = (source / "moment_initializer.py").read_bytes()
    manifest_bytes = (source / "frozen_method.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("source is not the exact reported frozen initializer")
    if manifest["sha256"]["moment_initializer.py"] != ORIGINAL_SHA256:
        raise ValueError("manifest does not name the reported frozen initializer")
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen dependency hash mismatch: {name}")
    text = original.decode("utf-8")
    anchor = "from scipy.optimize import minimize\n"
    if text.count(OLD) != 1 or text.count(anchor) != 1:
        raise ValueError("frozen source anchors do not match")
    text = text.replace(anchor, anchor + "from lbfgs_covariance import lbfgs_covariance_parts\n")
    amended = text.replace(OLD, NEW).encode("utf-8")
    helper = Path(__file__).with_name("lbfgs_covariance.py").read_bytes()
    hashes = {"moment_initializer.py": hashlib.sha256(amended).hexdigest(),
              "lbfgs_covariance.py": hashlib.sha256(helper).hexdigest()}
    manifest["sha256"].update(hashes)
    manifest.setdefault("amendments", []).append({
        "at": datetime.now(timezone.utc).isoformat(),
        "file": "moment_initializer.py",
        "previous_sha256": ORIGINAL_SHA256,
        "sha256": hashes["moment_initializer.py"],
        "helper_sha256": hashes["lbfgs_covariance.py"],
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "change": "Compute the same L-BFGS inverse diagonal and population block from "
                  "correction prefixes using SciPy vector products and inverse-BFGS "
                  "rank-two updates; no full-dimensional matrix. Equivalent in exact "
                  "arithmetic, with floating-point rounding differences. No model, "
                  "mode-search, BaM, stopping, chart, seed or tuning change. "
                  "Prepared without running or tuning on wren.",
    })
    output.mkdir(parents=True, exist_ok=False)
    (output / "moment_initializer.py").write_bytes(amended)
    (output / "lbfgs_covariance.py").write_bytes(helper)
    (output / "frozen_method.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_amendment(args.source, args.output), indent=2))
