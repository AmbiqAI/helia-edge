"""CorNET geometry against the paper's Table III and rolled/unrolled equivalence."""

import os
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
from pydantic import ValidationError

from helia_edge.models import CorNetParams, ModelSpec
from helia_edge.models import build as build_spec

# Table III, HR network: trainable parameters per layer (the dense head here is the 1-neuron HR output).
TABLE_III = {"conv0_conv": 1312, "conv1_conv": 40992, "lstm0": 82432, "lstm1": 131584, "hr": 129}


def build(unroll=False, params=CorNetParams(), batch_size=None):
    params = params.model_copy(update={"unroll": unroll})
    return build_spec(ModelSpec(params=params, input_shape=(1000, 1)), batch_size=batch_size)


def test_default_matches_paper_table_iii():
    model = build()
    assert model.name == "cornet"
    assert build_spec(ModelSpec(params=CorNetParams(), input_shape=(1000, 1)), name="hr").name == "hr"
    for name, count in TABLE_III.items():
        assert sum(int(np.prod(w.shape)) for w in model.get_layer(name).trainable_weights) == count
    assert model.get_layer("conv1_pool").output.shape[1] == 50
    assert model.get_layer("lstm0").output.shape == (None, 50, 128)
    assert model.get_layer("lstm1").output.shape == (None, 128)
    assert model.output_shape == (None, 1)


def test_stage_layer_order_follows_figure_6():
    model = build()
    names = [layer.name for layer in model.layers[1:]]
    stage = ["conv", "bn", "relu", "pool", "dropout"]
    assert names == [f"conv{i}_{part}" for i in range(2) for part in stage] + ["lstm0", "lstm1", "hr"]
    assert all(model.get_layer(name).recurrent_activation.__name__ == "sigmoid" for name in ("lstm0", "lstm1"))
    legacy = build(params=CorNetParams(recurrent_activation="hard_sigmoid"))
    assert legacy.get_layer("lstm0").recurrent_activation.__name__ == "hard_sigmoid"


def test_unrolled_equals_rolled():
    keras.utils.set_random_seed(3)
    rolled = build()
    unrolled = build(unroll=True)
    unrolled.set_weights(rolled.get_weights())
    x = np.random.default_rng(0).normal(size=(2, 1000, 1)).astype(np.float32)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(unrolled(x)), keras.ops.convert_to_numpy(rolled(x)), rtol=1e-5, atol=1e-6
    )


def test_invalid_configs_fail():
    for config in ({"filters": 0}, {"dropout": 1.0}, {"lstm_layers": 0}, {"width": 2}, {"name": "cornet"}):
        with pytest.raises(ValidationError):
            CorNetParams.model_validate(config)
    with pytest.raises(ValueError, match="too short"):
        build_spec(ModelSpec(params=CorNetParams(), input_shape=(100, 1)))
    with pytest.raises(ValueError, match="time, channels"):
        build_spec(ModelSpec(params=CorNetParams(), input_shape=(1000,)))


def test_keras_file_roundtrip(tmp_path):
    model = build(unroll=True, params=CorNetParams(lstm_units=16))
    path = tmp_path / "cornet.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    x = np.random.default_rng(1).normal(size=(1, 1000, 1)).astype(np.float32)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(model(x)), keras.ops.convert_to_numpy(loaded(x)))


@pytest.mark.skipif(keras.backend.backend() != "tensorflow", reason="LiteRT conversion needs TensorFlow")
@pytest.mark.parametrize("unroll", [False, True])
def test_litert_lowering_form(unroll):
    from tensorflow.lite.python import schema_py_generated as schema

    from helia_edge.export import ExportSpec, export_model

    model = build(unroll=unroll, params=CorNetParams(lstm_units=16), batch_size=1)
    content = export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")).content
    flat = schema.Model.GetRootAsModel(content, 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    ops = set()
    for g in range(flat.SubgraphsLength()):
        graph = flat.Subgraphs(g)
        for i in range(graph.OperatorsLength()):
            code = flat.OperatorCodes(graph.Operators(i).OpcodeIndex())
            ops.add(names[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())])
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" not in ops
    assert ("WHILE" in ops) is not unroll
    assert {"FULLY_CONNECTED", "LOGISTIC", "TANH", "MUL"} <= ops


def test_params_import_without_backends():
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'keras', 'tensorflow', 'torch', 'jax'}:
            raise AssertionError('config imported optional backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
from helia_edge.models import CorNetParams, ModelSpec
spec = ModelSpec(params=CorNetParams(unroll=True), input_shape=(1000, 1))
assert ModelSpec.model_validate_json(spec.model_dump_json()) == spec
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
