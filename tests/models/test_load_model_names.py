"""helia_edge.models.load_model on .keras files whose layer names contain '.' (earlier helia-edge naming).

The fixtures were saved by helia-edge main on TensorFlow, with their TF outputs; see
tests/fixtures/dotted_names/README.md.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from helia_edge.layers import LayerNormalization  # noqa: E402
from helia_edge.models import load_model  # noqa: E402
from helia_edge.models.utils import undot_layer_names, use_helia_layer_normalization  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dotted_names"
MODEL = FIXTURE / "efficientnetv2.keras"


TWO_OUTPUTS = FIXTURE / "two_outputs.keras"
TCN_LAYER_NORM = FIXTURE / "tcn_layer_norm.keras"
TORCH = keras.backend.backend() == "torch"


def saved_config(path=MODEL):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("config.json"))


def layer_names(model):
    return [layer.name for layer in model.layers]


def histories(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "keras_history":
                yield value[0]
            else:
                yield from histories(value)
    elif isinstance(node, list):
        for value in node:
            yield from histories(value)


def test_renaming_is_consistent_and_deterministic():
    config = saved_config()
    renamed, mapping = undot_layer_names(config)
    assert mapping and all("." in old and new == old.replace(".", "_") for old, new in mapping.items())
    layers = renamed["config"]["layers"]
    names = {layer["name"] for layer in layers}
    assert not any("." in name for name in names)
    assert all(layer["config"]["name"] == layer["name"] for layer in layers)
    assert set(histories(renamed)) <= names
    assert undot_layer_names(config) == (renamed, mapping)
    assert config == saved_config(), "the input config must not be modified"


@pytest.mark.parametrize("form", ["flat", "nested", "dict"])
def test_model_inputs_and_outputs_are_renamed(form):
    def ref(name):
        if form == "dict":
            return {name: [name, 0, 0]}
        return [[name, 0, 0]] if form == "nested" else [name, 0, 0]

    def tensor(name):
        return {"class_name": "__keras_tensor__", "config": {"keras_history": [name, 0, 0]}}

    config = {
        "class_name": "Functional",
        "config": {
            "name": "m",
            "layers": [
                {"class_name": "InputLayer", "name": "x.in", "config": {"name": "x.in"}, "inbound_nodes": []},
                {
                    "class_name": "Dense",
                    "name": "y.out",
                    "config": {"name": "y.out"},
                    "inbound_nodes": [{"args": [tensor("x.in")], "kwargs": {}}],
                },
            ],
            "input_layers": ref("x.in"),
            "output_layers": ref("y.out"),
        },
    }
    renamed, mapping = undot_layer_names(config)
    assert mapping == {"x.in": "x_in", "y.out": "y_out"}
    assert renamed["config"]["input_layers"] == ref("x_in")
    assert renamed["config"]["output_layers"] == ref("y_out")
    assert list(histories(renamed)) == ["x_in"]


def test_a_name_collision_is_refused():
    config = {
        "class_name": "Functional",
        "config": {
            "name": "m",
            "layers": [
                {"class_name": "Dense", "name": "a.b", "config": {"name": "a.b"}},
                {"class_name": "Dense", "name": "a_b", "config": {"name": "a_b"}},
            ],
        },
    }
    with pytest.raises(ValueError, match="'a.b' to 'a_b'.*collide"):
        undot_layer_names(config)


def test_tensorflow_loads_the_file_unchanged():
    if keras.backend.backend() != "tensorflow":
        pytest.skip("TensorFlow path")
    model = load_model(MODEL)
    assert any("." in name for name in layer_names(model))
    io = np.load(FIXTURE / "efficientnetv2_io.npz")
    for x, y in zip(io["inputs"], io["outputs"], strict=True):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(model(x, training=False)), y)


def test_torch_loads_the_dotted_file_with_the_tensorflow_outputs():
    if keras.backend.backend() != "torch":
        pytest.skip("Torch path")
    with pytest.raises(KeyError, match="parameter name can.+t contain"):
        keras.models.load_model(MODEL)
    model = load_model(MODEL)
    assert not any("." in name for name in layer_names(model))
    io = np.load(FIXTURE / "efficientnetv2_io.npz")
    for x, y in zip(io["inputs"], io["outputs"], strict=True):
        np.testing.assert_allclose(keras.ops.convert_to_numpy(model(x, training=False)), y, rtol=1e-5, atol=1e-6)


def test_compile_config_output_keys_are_renamed():
    renamed, mapping = undot_layer_names(saved_config(TWO_OUTPUTS))
    assert {"out.a", "out.b"} <= mapping.keys()
    compiled = json.dumps(renamed["compile_config"])
    assert '"out.a"' not in compiled and '"out_a"' in compiled and '"out_b"' in compiled


def test_keras_layer_normalization_entries_are_swapped():
    swapped, count = use_helia_layer_normalization(saved_config(TCN_LAYER_NORM))
    assert count > 0
    assert '"module": "keras.layers", "class_name": "LayerNormalization"' not in json.dumps(swapped)


def test_registration_finds_the_layer_normalization_class():
    code = (
        "import keras, helia_edge; helia_edge.register_keras_serializables(); "
        "cls = keras.saving.get_registered_object('helia_edge>LayerNormalization'); "
        "assert cls is not None and cls.__module__ == 'helia_edge.layers.normalization', cls"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_output_model_predicts_and_evaluates_on_every_backend():
    model = load_model(TWO_OUTPUTS)
    io = np.load(FIXTURE / "two_outputs_io.npz")
    names = ["out_a", "out_b"] if TORCH else ["out.a", "out.b"]
    predicted = model.predict(io["inputs"], verbose=0)
    np.testing.assert_allclose(predicted[names[0]], io["output_a"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(predicted[names[1]], io["output_b"], rtol=1e-5, atol=1e-6)
    result = model.evaluate(
        io["inputs"], {names[0]: io["target_a"], names[1]: io["target_b"]}, verbose=0, return_dict=True
    )
    assert np.isfinite(result["loss"])


def test_tcn_with_spatial_layer_norm_loads_with_the_tensorflow_outputs():
    model = load_model(TCN_LAYER_NORM)
    norms = [layer for layer in model.layers if isinstance(layer, keras.layers.LayerNormalization)]
    assert norms and all(isinstance(layer, LayerNormalization) for layer in norms) == TORCH
    io = np.load(FIXTURE / "tcn_layer_norm_io.npz")
    for x, y in zip(io["inputs"], io["outputs"], strict=True):
        np.testing.assert_allclose(keras.ops.convert_to_numpy(model(x, training=False)), y, rtol=1e-5, atol=1e-6)
