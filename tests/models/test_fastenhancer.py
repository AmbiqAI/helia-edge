"""FastEnhancer Keras graph against an independent NumPy reference of the ONNX semantics."""

import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from helia_edge.models import FastEnhancerModel, FastEnhancerParams
from helia_edge.models.fastenhancer import (
    FASTENHANCER_T_ONNX_NAMES,
    fastenhancer_t_onnx_weights,
    fastenhancer_weight_shapes,
    linear_filterbanks,
    load_fastenhancer_weights,
)

sys.path.insert(0, str(Path(__file__).parent))
from fastenhancer_reference import reference_frame  # noqa: E402

# Float32 Keras vs float64 reference: max abs error over the reference peak
# (at least 1). Correct graphs measure below 1e-6; unconverted PyTorch gate
# order, blocked qkv columns, swapped strided channels or a flipped upsampler
# kernel measure above 5e-3.
TOLERANCE = 1e-4

VARIANT = FastEnhancerParams(
    n_fft=64,
    channels=12,
    kernel_size=(4, 5, 3),
    stride=2,
    rnnformer={
        "num_blocks": 1,
        "channels": 12,
        "freq": 8,
        "num_heads": 3,
        "positional_embedding": False,
        "attn_bias": True,
        "post_act": True,
    },
    activation="relu",
    mask="sigmoid",
    input_compression=0.5,
    resnet=True,
)


def random_tensors(params, seed=0):
    """Fan-in scaled weights, small biases, so activations stay unsaturated."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, shape in fastenhancer_weight_shapes(params).items():
        if len(shape) == 1 or name.endswith((".B", ".pe")):
            std = 0.1
        elif name.endswith(".kernel"):
            std = shape[0] ** -0.5
        else:  # conv [out, in, k] and GRU [1, 3H, in]
            std = (shape[-1] * shape[1] if len(shape) == 3 and not name.endswith((".W", ".R")) else shape[-1]) ** -0.5
        tensors[name] = rng.normal(0.0, std, shape).astype(np.float32)
    return tensors


def run(params, step, spec, caches):
    outs, states = [], []
    for frame in range(spec.shape[2]):
        out, caches = step(spec[:, :, frame : frame + 1], caches)
        outs.append(np.asarray(out))
        states.append([np.asarray(c) for c in caches])
    return outs, states


def keras_step(model):
    def step(spec, caches):
        out = model([spec, *caches], training=False)
        return keras.ops.convert_to_numpy(out[0]), [keras.ops.convert_to_numpy(o) for o in out[1:]]

    return step


def max_error(model, params, tensors, frames=4, batch=2, seed=1):
    rng = np.random.default_rng(seed)
    spec = rng.normal(0.0, 1.0, (batch, params.spectral_bins, frames, 2)).astype(np.float32)
    rf = params.rnnformer
    caches = [np.zeros((batch, rf.freq, rf.channels), np.float32) for _ in range(rf.num_blocks)]
    got = run(params, keras_step(model), spec, caches)
    want = run(params, lambda s, c: reference_frame(params, tensors, s, c), spec, caches)

    def scaled(pairs):
        pairs = list(pairs)
        peak = max(np.abs(w).max() for _, w in pairs)
        return float(max(np.abs(g - w).max() for g, w in pairs) / max(peak, 1.0))

    out_error = scaled(zip(got[0], want[0]))
    state_error = scaled((g, w) for gs, ws in zip(got[1], want[1]) for g, w in zip(gs, ws))
    return out_error, state_error


def hydrated(params, seed=0):
    tensors = random_tensors(params, seed)
    model = FastEnhancerModel.model_from_params(params)
    load_fastenhancer_weights(model, params, tensors)
    return model, tensors


@pytest.mark.parametrize("params", [FastEnhancerParams(), VARIANT], ids=["t", "variant"])
def test_streaming_frames_match_reference(params):
    model, tensors = hydrated(params)
    out_error, state_error = max_error(model, params, tensors)
    assert out_error < TOLERANCE and state_error < TOLERANCE


def test_named_streaming_signature():
    model = FastEnhancerModel.model_from_params(FastEnhancerParams())
    assert [t.name for t in model.inputs] == ["spec_in", "cache_in_0", "cache_in_1"]
    assert [tuple(t.shape) for t in model.inputs] == [(None, 257, 1, 2), (None, 16, 20), (None, 16, 20)]
    assert model.output_names == ["spec_out", "cache_out_0", "cache_out_1"]
    assert [tuple(t.shape) for t in model.outputs] == [(None, 257, 1, 2), (None, 16, 20), (None, 16, 20)]


def test_carried_state_changes_output():
    params = FastEnhancerParams()
    model, _ = hydrated(params)
    spec = np.random.default_rng(2).normal(size=(1, 257, 1, 2)).astype(np.float32)
    zeros = [np.zeros((1, 16, 20), np.float32)] * 2
    carried = [np.full((1, 16, 20), 0.5, np.float32)] * 2
    first = keras.ops.convert_to_numpy(model([spec, *zeros])[0])
    second = keras.ops.convert_to_numpy(model([spec, *carried])[0])
    assert np.abs(first - second).max() > 1e-2
    assert first[:, -1].max() == 0 and first[:, -1].min() == 0


def test_hydration_rejects_mismatch_without_partial_load():
    params = FastEnhancerParams()
    model, tensors = hydrated(params)
    before = [w.copy() for w in model.get_weights()]
    changed = {name: value + 1 for name, value in tensors.items()}
    for bad in (
        {k: v for k, v in changed.items() if k != "rf_block.1.rnn.B"},
        {**changed, "rf_block.2.rnn.B": changed["rf_block.1.rnn.B"]},
        {**changed, "rf_block.0.rnn.W": changed["rf_block.0.rnn.W"][:, :, :-1]},
        {**changed, "enc_pre.0.bias": changed["enc_pre.0.bias"].astype(np.int32)},
    ):
        with pytest.raises(ValueError):
            load_fastenhancer_weights(model, params, bad)
        assert all(np.array_equal(a, b) for a, b in zip(before, model.get_weights()))
    other = FastEnhancerModel.model_from_params(VARIANT)
    with pytest.raises(ValueError, match="does not match params"):
        load_fastenhancer_weights(other, params, tensors)


def test_release_names_map_to_module_paths():
    expected = fastenhancer_weight_shapes(FastEnhancerParams())
    release = {onnx: np.zeros(expected[key], np.float32) for onnx, key in FASTENHANCER_T_ONNX_NAMES.items()}
    release.update(
        {
            key: np.zeros(shape, np.float32)
            for key, shape in expected.items()
            if key not in FASTENHANCER_T_ONNX_NAMES.values()
        }
    )
    release["/Constant_6_output_0"] = np.float32(1e-5)
    assert set(fastenhancer_t_onnx_weights(release)) == set(expected)
    with pytest.raises(ValueError, match="unrecognized"):
        fastenhancer_t_onnx_weights({**release, "onnx::MatMul_999": np.zeros((2, 2), np.float32)})


def test_fixed_filterbanks_are_frozen_normalized_projections():
    pre, post = linear_filterbanks(64, 16)
    assert pre.shape == (64, 16) and post.shape == (16, 64)
    np.testing.assert_allclose(pre.sum(axis=0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(post.sum(axis=0), 1.0, rtol=1e-6)
    # Values of the onnx-vd-v1.0.0 fixed projections; the later upstream formula differs.
    np.testing.assert_allclose(pre[:5, 0], [0.36670548, 0.29802096, 0.2048894, 0.11175784, 0.01862631], atol=1e-6)
    np.testing.assert_allclose(post[0, :3], [1.0, 0.7619048, 0.52380955], atol=1e-6)
    model = FastEnhancerModel.model_from_params(FastEnhancerParams())
    assert not model.get_layer("rf_pre_proj").trainable
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(model.get_layer("rf_pre_proj").kernel), pre)


def test_keras_file_roundtrip(tmp_path):
    params = VARIANT
    model, _ = hydrated(params)
    path = tmp_path / "fastenhancer.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    rng = np.random.default_rng(3)
    inputs = [rng.normal(size=(1, *t.shape[1:])).astype(np.float32) for t in model.inputs]
    for a, b in zip(model(inputs), loaded(inputs)):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))
