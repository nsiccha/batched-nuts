"""Build an immutable amendment of a frozen research moment initializer.

The research run's directory is an input, never an output. For one frozen
version this emits the amended initializer, the helper modules it imports and
the updated manifest into a new directory for review/distribution; all other
files named by the manifest remain the original frozen dependencies.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

IMPORT_ANCHOR = "from scipy.optimize import minimize\n"

# Each amendment replaces one exact passage and imports one helper from this directory.
COVARIANCE = dict(
    helper="lbfgs_covariance.py",
    imported="from lbfgs_covariance import lbfgs_covariance_parts\n",
    old="""            inverse = np.asarray(res.hess_inv.todense())
            inverse = .5 * (inverse + inverse.T)
            start_var = np.diag(inverse).copy()
            if not (np.isfinite(inverse).all() and (start_var > 0).all()):
                raise FloatingPointError("mode search returned an unusable inverse-Hessian approximation")
            state = (jnp.asarray(res.x), jnp.asarray(start_var),
                     jnp.asarray(inverse[np.ix_(self.pop, self.pop)]), state[3])""",  # noqa: E501
    new="""            start_var, start_cov = lbfgs_covariance_parts(res.hess_inv, self.pop)
            if not (np.isfinite(start_var).all() and np.isfinite(start_cov).all()
                    and (start_var > 0).all()):
                raise FloatingPointError("mode search returned an unusable inverse-Hessian approximation")
            state = (jnp.asarray(res.x), jnp.asarray(start_var),
                     jnp.asarray(start_cov), state[3])""",  # noqa: E501
    change=(
        "Compute the same L-BFGS inverse diagonal and population block from "
        "correction prefixes using SciPy vector products and inverse-BFGS "
        "rank-two updates; no full-dimensional matrix. Equivalent in exact "
        "arithmetic, with floating-point rounding differences. No model, "
        "mode-search, BaM, stopping, chart, seed or tuning change. "
        "Prepared without running or tuning on wren."
    ),
)
# Versions 2 and 3 keep the mode search only as the optional locate=True start.
COVARIANCE_OPTIONAL = dict(
    COVARIANCE,
    change=(
        COVARIANCE["change"]
        + " The frozen configuration has locate=False and never runs this mode search."
    ),
)
GEOMETRY = dict(
    helper="population_jacobian.py",
    imported="from population_jacobian import population_jacobian\n",
    old=(
        "            jac = jax.jacfwd(lambda x: prior(x)[1])(mean)[..., pop]"
        "          # (J, b, p)\n"
    ),
    new=(
        "            jac = population_jacobian(lambda x: prior(x)[1], mean, pop)"
        "      # (J, b, p)\n"
    ),
    change=(
        "Differentiate each effect's prior log scale with respect to the "
        "population coordinates only, instead of with respect to every "
        "coordinate followed by keeping the population columns. These are "
        "the same forward-mode partial derivatives, so the low-scale "
        "centering formula and z_low are unchanged; no array has two axes "
        "that grow with the dimension. No model, BaM, tempering, stopping, "
        "chart, seed or tuning change. Prepared without running a fit or "
        "tuning on wren."
    ),
)

# The exact frozen sources this builder amends, by version of the method.
FROZEN = {
    1: dict(
        file="moment_initializer.py",
        manifest="frozen_method.json",
        sha256="fc3609757f647a4a9ecadd9d1aeac99d18c45dd58815f5e0174390dbfdf3c216",
        amendments=(COVARIANCE,),
    ),
    2: dict(
        file="moment_initializer_v2.py",
        manifest="frozen_method_v2.json",
        sha256="07626d760f5720564e56697b2b17e77dc2f325557c41c3a3a356522851fb232f",
        amendments=(GEOMETRY, COVARIANCE_OPTIONAL),
    ),
    3: dict(
        file="moment_initializer_v3.py",
        manifest="frozen_method_v3.json",
        sha256="ced682b9fbd2e9bb251f6322128e16c5b14e0616f191dbc8732d97782cad86f7",
        amendments=(GEOMETRY, COVARIANCE_OPTIONAL),
    ),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build_amendment(source, output, version=1):
    frozen = FROZEN[version]
    source, output = Path(source), Path(output)
    original = (source / frozen["file"]).read_bytes()
    manifest_bytes = (source / frozen["manifest"]).read_bytes()
    manifest = json.loads(manifest_bytes)
    if sha256(original) != frozen["sha256"]:
        raise ValueError("source is not the exact reported frozen initializer")
    if manifest["sha256"][frozen["file"]] != frozen["sha256"]:
        raise ValueError("manifest does not name the reported frozen initializer")
    for name, expected in manifest["sha256"].items():
        if sha256((source / name).read_bytes()) != expected:
            raise ValueError(f"frozen dependency hash mismatch: {name}")
    text = original.decode("utf-8")
    if text.count(IMPORT_ANCHOR) != 1 or any(
        text.count(amendment["old"]) != 1 for amendment in frozen["amendments"]
    ):
        raise ValueError("frozen source anchors do not match")
    at = datetime.now(UTC).isoformat()
    helpers, previous = {}, frozen["sha256"]
    for amendment in frozen["amendments"]:
        text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + amendment["imported"])
        text = text.replace(amendment["old"], amendment["new"])
        helper = Path(__file__).with_name(amendment["helper"]).read_bytes()
        helpers[amendment["helper"]] = helper
        current = sha256(text.encode("utf-8"))
        # One entry per replaced passage; consecutive entries chain by hash.
        manifest.setdefault("amendments", []).append(
            {
                "at": at,
                "file": frozen["file"],
                "previous_sha256": previous,
                "sha256": current,
                "helper_sha256": sha256(helper),
                "source_manifest_sha256": sha256(manifest_bytes),
                "change": amendment["change"],
            }
        )
        previous = current
    amended = text.encode("utf-8")
    hashes = {
        frozen["file"]: sha256(amended),
        **{name: sha256(helper) for name, helper in helpers.items()},
    }
    manifest["sha256"].update(hashes)
    output.mkdir(parents=True, exist_ok=False)
    (output / frozen["file"]).write_bytes(amended)
    for name, helper in helpers.items():
        (output / name).write_bytes(helper)
    (output / frozen["manifest"]).write_text(json.dumps(manifest, indent=2) + "\n")
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--version",
        type=int,
        choices=sorted(FROZEN),
        default=1,
        help="frozen version of the method to amend (default: 1)",
    )
    args = parser.parse_args()
    print(json.dumps(build_amendment(args.source, args.output, args.version), indent=2))
