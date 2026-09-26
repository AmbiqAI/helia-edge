"""Compare constructors with independently extracted pinned reference configs."""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

from helia_edge.models import mlperf_tiny_ad, mlperf_tiny_kws, mlperf_tiny_resnet, mlperf_tiny_vww

BUILDERS = {"kws": mlperf_tiny_kws, "vww": mlperf_tiny_vww, "resnet": mlperf_tiny_resnet, "ad": mlperf_tiny_ad}
REFERENCE = json.loads((Path(__file__).parents[1] / "fixtures/mlperf-tiny-reference.json").read_text())


def normalize(value):
    if isinstance(value, dict):
        if "class_name" in value and "config" in value:
            return {"class_name": value["class_name"], "config": normalize(value["config"])}
        return {k: normalize(v) for k, v in value.items()
                if not (k in {"input_axes", "output_axes"} and v is None)}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    if isinstance(value, float):
        return round(value, 7)
    return value


def assert_reference_architecture(model, name):
    reference = REFERENCE[name]
    specs = {layer["id"]: layer for layer in reference["layers"]}
    fields = {}
    for layer in reference["layers"]:
        fields.setdefault(layer["class"], set()).update(layer["config"])

    def expected(node_id):
        node = specs[node_id]
        return node["class"], node["config"], tuple(expected(i) for i in node["inputs"])

    def actual(tensor):
        layer = tensor._keras_history.operation
        cls = type(layer).__name__
        assert cls in fields, f"Unexpected layer {cls}"
        config = layer.get_config()
        params = normalize({key: config[key] if key in config else getattr(layer, key) for key in fields[cls]})
        inputs = [] if cls == "InputLayer" else layer._inbound_nodes[0].input_tensors
        return cls, params, tuple(actual(i) for i in inputs)

    assert len(model.layers) == len(reference["layers"])
    assert model.input.dtype == "float32" and model.output.dtype == "float32"
    assert tuple(actual(t) for t in model.outputs) == tuple(expected(i) for i in reference["outputs"])


@pytest.mark.parametrize("name", BUILDERS)
def test_full_reference_layer_semantics_and_connectivity(name):
    keras.backend.clear_session()
    model = BUILDERS[name]()
    assert_reference_architecture(model, name)


@pytest.mark.parametrize("name", BUILDERS)
def test_seed_reference_output_and_serialization(name, tmp_path):
    keras.backend.clear_session()
    keras.utils.set_random_seed(20260926)
    model = BUILDERS[name]()
    shape = (1, *model.input_shape[1:])
    x = np.random.default_rng(30).uniform(-1, 1, shape).astype(np.float32)
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    assert np.isfinite(expected).all()
    if name != "ad":
        np.testing.assert_allclose(expected.sum(axis=-1), 1, atol=1e-6)
    path = tmp_path / f"{name}.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_array_equal(expected, keras.ops.convert_to_numpy(restored(x, training=False)))
    keras.backend.clear_session()
    keras.utils.set_random_seed(20260926)
    again = BUILDERS[name]()
    for left, right in zip(model.get_weights(), again.get_weights(), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("name,kind", [("kws", "pool"), ("vww", "relu6"), ("resnet", "residual"), ("ad", "output")])
def test_reference_check_rejects_plausible_architecture_mutations(name, kind):
    original = BUILDERS[name]()

    def clone(layer):
        config = layer.get_config()
        if kind == "pool" and isinstance(layer, keras.layers.AveragePooling2D):
            config["pool_size"] = (24, 5)
        elif kind == "relu6" and isinstance(layer, keras.layers.Activation):
            config["activation"] = "relu6"
        elif kind == "residual" and isinstance(layer, keras.layers.Add):
            return keras.layers.Lambda(lambda tensors: tensors[1], name=layer.name)
        elif kind == "output" and layer.name == "reconstruction":
            config["activation"] = "sigmoid"
        return type(layer).from_config(config)

    mutant = keras.models.clone_model(original, clone_function=clone)
    with pytest.raises(AssertionError):
        assert_reference_architecture(mutant, name)
