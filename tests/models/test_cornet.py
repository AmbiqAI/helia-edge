"""CorNET geometry against the paper's Table III and rolled/unrolled equivalence."""

import os
from pathlib import Path
import subprocess
import sys

import keras
import numpy as np
import pytest
from pydantic import ValidationError

from helia_edge.models import CorNetModel, CorNetParams

# Table III, HR network: trainable parameters per layer (the dense head here is the 1-neuron HR output).
TABLE_III = {"conv0_conv": 1312, "conv1_conv": 40992, "lstm0": 82432, "lstm1": 131584, "hr": 129}


def build(unroll=False, params=CorNetParams()):
    return CorNetModel.model_from_params(keras.Input((1000, 1)), params, unroll=unroll)


def test_default_matches_paper_table_iii():
    model = build()
    for name, count in TABLE_III.items():
        assert sum(int(np.prod(w.shape)) for w in model.get_layer(name).trainable_weights) == count
    assert model.get_layer("conv1_pool").output.shape[1] == 50
    assert model.get_layer("lstm0").output.shape == (None, 50, 128)
    assert model.get_layer("lstm1").output.shape == (None, 128)
    assert model.output_shape == (None, 1)


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
    for config in ({"filters": 0}, {"dropout": 1.0}, {"lstm_layers": 0}, {"width": 2}, {"filters": "32"}):
        with pytest.raises(ValidationError):
            CorNetParams.from_config(config)
    with pytest.raises(ValueError, match="too short"):
        CorNetModel.model_from_params(keras.Input((100, 1)), CorNetParams())
    with pytest.raises(TypeError):
        CorNetModel.model_from_params(keras.Input((1000, 1)), {"filters": 32})


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

    from helia_edge.converters.litert import ConversionType, LiteRTKerasConverter, QuantizationType

    params = CorNetParams(lstm_units=16)
    model = CorNetModel.model_from_params(keras.Input((1000, 1), batch_size=1), params, unroll=unroll)
    converter = LiteRTKerasConverter(model)
    try:
        content = converter.convert(quantization=QuantizationType.FP32, mode=ConversionType.CONCRETE)
    finally:
        converter.cleanup()
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
from helia_edge.models import CorNetParams
params = CorNetParams()
assert CorNetParams.from_config(params.get_config()) == params
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
