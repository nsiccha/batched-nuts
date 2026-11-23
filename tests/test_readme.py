"""The usage example of the README runs and does what the README says."""

import re
from pathlib import Path

import numpy as np


def test_usage_example():
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    example = re.search(r"```python\n(.*?)```", readme, re.DOTALL).group(1)
    names = {}
    exec(example, names)
    draws = names["draws"]
    assert draws.position.shape == (200, 250, 4)
    assert isinstance(draws.position, np.ndarray)
    # every chain samples its own target: a normal distribution around its dataset's mean
    means = np.asarray(names["data"]["mean"])[np.asarray(names["jobs"].dataset)]
    error = draws.position.mean(axis=1) - means
    assert np.abs(error).max() < 1.0
    assert np.abs(error.mean(axis=0)).max() < 0.05
