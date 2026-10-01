"""helia_edge.models.load_model on .keras files whose layer names contain '.' (earlier helia-edge naming).

The fixture is a seeded EfficientNetV2 saved by helia-edge main a677c85b on TensorFlow, with its TF outputs;
tests/fixtures/dotted_names/README.md says how to regenerate it.
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")

from helia_edge.models import load_model  # noqa: E402
from helia_edge.models.utils import undot_layer_names  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dotted_names"
MODEL = FIXTURE / "efficientnetv2.keras"


def saved_config():
    with zipfile.ZipFile(MODEL) as archive:
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


@pytest.mark.parametrize("nested", [False, True])
def test_model_inputs_and_outputs_are_renamed(nested):
    def ref(name):
        return [[name, 0, 0]] if nested else [name, 0, 0]

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
