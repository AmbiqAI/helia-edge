"""TimePPG against an independent NumPy forward pass in the upstream NCW layout."""

import os
from pathlib import Path
import subprocess
import sys

import keras
import numpy as np
import pytest
from pydantic import ValidationError

from helia_edge.models import TIMEPPG_PRESETS, TimePPGModel, TimePPGParams


def upstream_params(channels, c_in=4, time=256):
    """Trainable parameters of the upstream PyTorch model (BatchNorm counts gamma and beta)."""
    kernels = (3, 3, 5, 3, 3, 5, 3, 3, 5)
    widths, total = (c_in, *channels[:9]), 0
    for k, c0, c1 in zip(kernels, widths[:-1], widths[1:]):
        total += c0 * c1 * k + 2 * c1
    features = channels[8] * (time // 64)
    total += features * channels[9] + 2 * channels[9]
    total += channels[9] * channels[10] + 2 * channels[10]
    return total + channels[10] + 1


def conv1d(x, weight, stride=1, dilation=1, pad=0):
    """PyTorch Conv1d on [N, C, T] with a Keras [K, Cin, Cout] kernel, no bias."""
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    k = weight.shape[0]
    length = (x.shape[2] - dilation * (k - 1) - 1) // stride + 1
    out = np.zeros((x.shape[0], weight.shape[2], length))
    for j in range(k):
        taps = x[:, :, j * dilation : j * dilation + stride * (length - 1) + 1 : stride]
        out += np.einsum("nct,co->not", taps, weight[j])
    return out


def batchnorm(x, layer, axis):
    gamma, beta, mean, var = (np.asarray(w, np.float64) for w in layer.get_weights())
    shape = [1] * x.ndim
    shape[axis] = -1
    return (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + 1e-5) * gamma.reshape(shape) + beta.reshape(shape)


def reference(model, params, x):
    """Upstream forward: TempConvBlock conv-BN-ReLU6; ConvBlock conv-AvgPool-BN-ReLU6; NCW flatten."""
    relu6 = lambda v: np.clip(v, 0, 6)  # noqa: E731
    w = lambda name: np.asarray(model.get_layer(name).get_weights()[0], np.float64)  # noqa: E731
    y = np.transpose(x, (0, 2, 1)).astype(np.float64)
    temporal = {"tcb00": 2, "tcb01": 2, "tcb10": 4, "tcb11": 4, "tcb20": 8, "tcb21": 8}
    conv = {"cb0": (1, 2), "cb1": (2, 2), "cb2": (4, 4)}
    for stage in range(3):
        for name in (f"tcb{stage}0", f"tcb{stage}1"):
            dilation = temporal[name]
            y = conv1d(y, w(f"{name}_conv"), dilation=dilation, pad=dilation)
            y = relu6(batchnorm(y, model.get_layer(f"{name}_bn"), 1))
        stride, pad = conv[f"cb{stage}"]
        y = conv1d(y, w(f"cb{stage}_conv"), stride=stride, pad=pad)
        y = y[:, :, : y.shape[2] // 2 * 2].reshape(y.shape[0], y.shape[1], -1, 2).mean(-1)
        y = relu6(batchnorm(y, model.get_layer(f"cb{stage}_bn"), 1))
    y = y.reshape(y.shape[0], -1)
    for index in range(2):
        y = relu6(batchnorm(y @ w(f"regr{index}_dense"), model.get_layer(f"regr{index}_bn"), 1))
    kernel, bias = model.get_layer("out_neuron").get_weights()
    return y @ kernel + bias


def randomized(params, seed=0):
    keras.utils.set_random_seed(seed)
    model = TimePPGModel.model_from_params(keras.Input((256, 4)), params)
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            c = layer.get_weights()[0].shape
            layer.set_weights([rng.uniform(0.5, 1.5, c), rng.normal(0, 0.1, c), rng.normal(0, 0.1, c), rng.uniform(0.5, 1.5, c)])
    return model


@pytest.mark.parametrize("preset", sorted(TIMEPPG_PRESETS))
def test_presets_match_upstream_parameter_count_and_geometry(preset):
    params = TIMEPPG_PRESETS[preset]
    model = TimePPGModel.model_from_params(keras.Input((256, 4)), params)
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    assert trainable == upstream_params(params.channels)
    assert model.output_shape == (None, 1)
    assert [model.get_layer(f"cb{i}_pool").output.shape[1] for i in range(3)] == [128, 32, 4]
    dilations = [model.get_layer(f"tcb{s}{i}_conv").dilation_rate[0] for s in range(3) for i in range(2)]
    assert dilations == [2, 2, 4, 4, 8, 8]
    assert all(model.get_layer(f"tcb{s}{i}_conv").kernel_size == (3,) for s in range(3) for i in range(2))
    assert [model.get_layer(f"cb{i}_conv").strides[0] for i in range(3)] == [1, 2, 4]


def test_medium_preset_size():
    assert upstream_params(TIMEPPG_PRESETS["timeppg_medium"].channels) == 40542


@pytest.mark.parametrize("preset", ["timeppg_medium", "timeppg_small"])
def test_forward_matches_upstream_layout(preset):
    params = TIMEPPG_PRESETS[preset]
    model = randomized(params)
    x = np.random.default_rng(1).normal(size=(3, 256, 4)).astype(np.float32)
    actual = keras.ops.convert_to_numpy(model(x, training=False))
    np.testing.assert_allclose(actual, reference(model, params, x), rtol=1e-4, atol=1e-4)


def test_invalid_configs_fail():
    for config in ({"channels": [1] * 10}, {"channels": [0] + [1] * 10}, {"width": 2}, {"channels": ["8"] * 11}):
        with pytest.raises(ValidationError):
            TimePPGParams.from_config(config)
    with pytest.raises(ValueError, match="time length"):
        TimePPGModel.model_from_params(keras.Input((32, 4)), TimePPGParams())


def test_keras_file_roundtrip(tmp_path):
    model = randomized(TIMEPPG_PRESETS["timeppg_small"])
    path = tmp_path / "timeppg.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    x = np.random.default_rng(2).normal(size=(1, 256, 4)).astype(np.float32)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(model(x)), keras.ops.convert_to_numpy(loaded(x)))


def test_params_import_without_backends():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'keras', 'tensorflow', 'torch', 'jax'}:
            raise AssertionError('config imported optional backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
from helia_edge.models import TIMEPPG_PRESETS, TimePPGParams
params = TIMEPPG_PRESETS['timeppg_medium']
assert TimePPGParams.from_config(params.get_config()) == params
'''
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root,
                   env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True)
