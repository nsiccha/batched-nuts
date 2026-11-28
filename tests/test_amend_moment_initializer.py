"""The amendment builder: exact passages replaced, hashes chained, inputs untouched."""

import hashlib
import json
import textwrap

import amend_moment_initializer as amend
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from population_jacobian import population_jacobian

FILE, MANIFEST = "moment_initializer_v9.py", "frozen_method_v9.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def research(tmp_path, monkeypatch):
    """A stand-in research directory whose initializer holds both amended passages."""
    source = tmp_path / "moments"
    (source / "vendor").mkdir(parents=True)
    (source / FILE).write_text(
        "import jax\n" + amend.IMPORT_ANCHOR + "\n\nclass MomentInitializer:\n"
        "    def information(mean, cov):\n"
        + amend.GEOMETRY["old"]
        + "            return jac\n\n    def fit(self, res, state):\n"
        + amend.COVARIANCE["old"]
        + "\n            return state\n"
    )
    (source / "vendor" / "update.py").write_text("lam = 1.0\n")
    hashes = {
        FILE: digest(source / FILE),
        "vendor/update.py": digest(source / "vendor/update.py"),
    }
    (source / MANIFEST).write_text(json.dumps({"version": 9, "sha256": hashes}))
    monkeypatch.setitem(
        amend.FROZEN,
        9,
        dict(
            file=FILE,
            manifest=MANIFEST,
            sha256=hashes[FILE],
            amendments=(amend.GEOMETRY, amend.COVARIANCE_OPTIONAL),
        ),
    )
    return source


def test_replaces_each_passage_and_chains_the_hashes(research, tmp_path):
    before = {path: path.read_bytes() for path in research.rglob("*") if path.is_file()}
    original = digest(research / FILE)
    output = tmp_path / "amended"
    hashes = amend.build_amendment(research, output, version=9)

    assert {
        path: path.read_bytes() for path in research.rglob("*") if path.is_file()
    } == before
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [FILE, MANIFEST, "lbfgs_covariance.py", "population_jacobian.py"]
    )
    text = (output / FILE).read_text()
    for amendment in (amend.GEOMETRY, amend.COVARIANCE):
        assert amendment["old"] not in text
        assert text.count(amendment["new"]) == 1
        assert text.count(amendment["imported"]) == 1
    assert "jacfwd" not in text and "todense" not in text

    manifest = json.loads((output / MANIFEST).read_text())
    for name in hashes:
        assert manifest["sha256"][name] == hashes[name] == digest(output / name)
    assert manifest["sha256"]["vendor/update.py"] == digest(research / "vendor/update.py")
    steps = manifest["amendments"]
    assert [step["helper_sha256"] for step in steps] == [
        hashes["population_jacobian.py"],
        hashes["lbfgs_covariance.py"],
    ]
    assert steps[0]["previous_sha256"] == original
    assert steps[1]["previous_sha256"] == steps[0]["sha256"]
    assert steps[1]["sha256"] == hashes[FILE]
    assert all(
        step["source_manifest_sha256"] == digest(research / MANIFEST) for step in steps
    )


def test_refuses_a_source_other_than_the_frozen_one(research, tmp_path):
    (research / FILE).write_text((research / FILE).read_text() + "# edited\n")
    with pytest.raises(ValueError, match="exact reported frozen initializer"):
        amend.build_amendment(research, tmp_path / "amended", version=9)
    assert not (tmp_path / "amended").exists()


def test_refuses_a_changed_dependency(research, tmp_path):
    (research / "vendor" / "update.py").write_text("lam = 2.0\n")
    with pytest.raises(ValueError, match="vendor/update.py"):
        amend.build_amendment(research, tmp_path / "amended", version=9)


def test_never_writes_into_an_existing_directory(research, tmp_path):
    output = tmp_path / "amended"
    output.mkdir()
    with pytest.raises(FileExistsError):
        amend.build_amendment(research, output, version=9)
    assert list(output.iterdir()) == []


def test_replaced_line_and_its_replacement_compute_the_same_jacobian():
    rng = np.random.default_rng(3)
    pop = np.array([7, 0, 3])
    weights = jnp.asarray(rng.normal(size=(5, 3)))
    mean = jnp.asarray(rng.normal(size=13))

    def prior(x):
        return jnp.zeros(5), jnp.tanh(weights @ x[pop]) + 0.2 * x[8:13] ** 2

    results = []
    for line in (amend.GEOMETRY["old"], amend.GEOMETRY["new"]):
        scope = dict(
            jax=jax,
            prior=prior,
            mean=mean,
            pop=pop,
            population_jacobian=population_jacobian,
        )
        exec(textwrap.dedent(line), scope)
        results.append(scope["jac"])
    assert results[0].shape == (5, 3)
    np.testing.assert_allclose(results[1], results[0], rtol=1e-13, atol=1e-15)
