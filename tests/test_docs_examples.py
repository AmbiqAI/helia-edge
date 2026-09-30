"""Execute authored introductory examples against the selected Keras backend."""

import json
import os
import re
from pathlib import Path

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


def test_generator_walkthrough():
    """The finite reader example preserves all records and the partial batch."""
    if os.environ.get("KERAS_BACKEND", "tensorflow") != "tensorflow":
        pytest.skip("The input-pipeline helper is TensorFlow-specific")
    pytest.importorskip("tensorflow")
    import numpy as np

    source = (ROOT / "astro-site/src/content/docs/guide/input-pipeline.mdx").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert len(blocks) == 3
    namespace = {}
    for block in blocks:
        exec(compile(block, "input-pipeline.mdx", "exec"), namespace)
    batches = namespace["batches"]
    assert [batch.shape for batch in batches] == [(2, 8, 1), (1, 8, 1)]
    np.testing.assert_array_equal(np.concatenate(batches)[:, 0, 0], [0, 1, 2])
    partitioned = list(namespace["partitioned"].as_numpy_iterator())
    np.testing.assert_array_equal(np.stack(partitioned)[:, 0, 0], [0, 1, 2])
    assert len(namespace["preview"]) == 6


def test_preprocessing_progression():
    """Composed training noise leaves the documented inference path deterministic."""
    import keras
    import numpy as np

    source = (ROOT / "astro-site/src/content/docs/guide/preprocessing.mdx").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert len(blocks) == 2
    namespace = {}
    for block in blocks:
        exec(compile(block, "preprocessing.mdx", "exec"), namespace)
    inference = keras.ops.convert_to_numpy(namespace["inference_inputs"])
    training = keras.ops.convert_to_numpy(namespace["training_inputs"])
    np.testing.assert_allclose(inference, 0.5, rtol=1e-5)
    assert inference.shape == training.shape == (2, 128, 1)
    assert not np.array_equal(inference, training)
