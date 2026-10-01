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
        block = block.replace(
            'os.environ["KERAS_BACKEND"] = "tensorflow"', f'os.environ["KERAS_BACKEND"] = "{backend}"'
        )
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
    source = source.split("## Build a finite dataset", 1)[1]
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


def test_grain_input_pipeline_example():
    """The Grain example in the input-pipeline guide prints the documented batch shapes."""
    pytest.importorskip("grain")
    source = (ROOT / "astro-site/src/content/docs/guide/input-pipeline.mdx").read_text()
    section = source.split("## Read records with Grain", 1)[1]
    (block,) = re.findall(r"```python\n(.*?)```", section.split("\n## ", 1)[0], re.DOTALL)
    namespace = {}
    exec(compile(block.replace("print(", "shapes = ("), "input-pipeline.mdx", "exec"), namespace)
    assert namespace["shapes"] == [(4, 8)] * 5


def test_export_examples(tmp_path, monkeypatch):
    """The export guide and the landing export step run with export_model on TensorFlow."""
    if os.environ.get("KERAS_BACKEND", "tensorflow") != "tensorflow":
        pytest.skip("LiteRT export needs the TensorFlow backend")
    pytest.importorskip("ai_edge_litert")
    import keras
    import numpy as np

    monkeypatch.chdir(tmp_path)
    model = keras.Sequential([keras.Input((8,)), keras.layers.Dense(2)])
    representative_x = np.random.default_rng(0).normal(size=(16, 8))
    guide = (ROOT / "astro-site/src/content/docs/guide/export.mdx").read_text()
    section = guide.split("## Export and inspect", 1)[1].split("\n## ", 1)[0]
    (block,) = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    namespace = {"model": model, "representative_x": representative_x}
    exec(compile(block, "export.mdx", "exec"), namespace)
    assert namespace["predictions"].shape == (16, 2)
    assert (tmp_path / "model.tflite").read_bytes() == namespace["result"].content
    steps = json.loads((ROOT / "astro-site/src/data/workbench.json").read_text())
    (export_step,) = [step for step in steps if step["guide"] == "export"]
    landing = {"model": model, "representative_x": representative_x}
    exec(compile(export_step["code"], "workbench.json", "exec"), landing)
    assert landing["result"].sha256 == namespace["result"].sha256


def test_streaming_export_example():
    """The export guide's streaming example runs and keeps the carried state's value."""
    if os.environ.get("KERAS_BACKEND", "tensorflow") != "tensorflow":
        pytest.skip("LiteRT export needs the TensorFlow backend")
    pytest.importorskip("ai_edge_litert")
    import numpy as np

    guide = (ROOT / "astro-site/src/content/docs/guide/export.mdx").read_text()
    section = guide.split("## Streaming models", 1)[1].split("\n## ", 1)[0]
    (block,) = re.findall(r"```python\n(.*?)```", section, re.DOTALL)
    namespace = {}
    exec(compile(block, "export.mdx", "exec"), namespace)
    fed, outputs = namespace["fed"], namespace["outputs"]
    assert namespace["probabilities"].shape == (50, 1, 1)
    np.testing.assert_array_equal(fed["state_in_1"][1:], outputs["state_out_1"][:-1])
    result = namespace["result"]
    for k in (0, 1):
        (state_in,) = [t for t in result.inputs if t.pair == k]
        (state_out,) = [t for t in result.outputs if t.pair == k]
        assert (state_in.scale, state_in.zero_point) == (state_out.scale, state_out.zero_point)
