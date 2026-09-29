"""Execute authored introductory examples against the selected Keras backend."""

import json
import os
from pathlib import Path
import re

import pytest

pytest.importorskip("keras")

ROOT = Path(__file__).resolve().parents[1]


def test_first_model_walkthrough(tmp_path, monkeypatch):
    """The published walkthrough must construct and save its declared artifacts."""
    monkeypatch.chdir(tmp_path)
    source = (ROOT / "astro-site/src/content/docs/getting-started/first-model.mdx").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert len(blocks) == 3
    namespace = {}
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    for block in blocks:
        block = block.replace('os.environ["KERAS_BACKEND"] = "tensorflow"',
                              f'os.environ["KERAS_BACKEND"] = "{backend}"')
        exec(compile(block, "first-model.mdx", "exec"), namespace)
    assert namespace["model"].output_shape == (1, 240, 2)
    assert (tmp_path / "model.keras").stat().st_size > 0
    assert json.loads((tmp_path / "architecture.json").read_text())


def test_landing_preprocessing_example():
    """The displayed transform produces the documented normalization."""
    import keras
    import numpy as np

    steps = json.loads((ROOT / "astro-site/src/data/workbench.json").read_text())
    namespace = {}
    exec(compile(steps[0]["code"], "workbench.json", "exec"), namespace)
    output = keras.ops.convert_to_numpy(namespace["inputs"])
    assert output.shape == (2, 128, 1)
    np.testing.assert_allclose(output, 0.5, rtol=1e-5)
