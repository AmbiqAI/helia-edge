"""FastEnhancer Keras graph against an independent NumPy reference of the ONNX semantics."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

from helia_edge.importers import SourcePin, import_weights
from helia_edge.models import FastEnhancerParams
from helia_edge.models.fastenhancer import build, fastenhancer_weight_shapes, linear_filterbanks
from helia_edge.models.fastenhancer_params import FASTENHANCER_T_ONNX, fastenhancer_mapping

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


def pinned(path, tensors, write_safetensors):
    write_safetensors(path, tensors)
    return SourcePin(uri=path.as_uri(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), format="safetensors")


@pytest.fixture
def hydrated(tmp_path, write_safetensors):
    """Build ``params`` and import random module-path tensors through ``fastenhancer_mapping``."""

    def hydrate(params, seed=0):
        tensors = random_tensors(params, seed)
        path = tmp_path / f"fastenhancer-{seed}.safetensors"
        model = build(params)
        import_weights(model, fastenhancer_mapping(params, "random", pinned(path, tensors, write_safetensors)), path)
        return model, tensors

    return hydrate


@pytest.mark.parametrize("params", [FastEnhancerParams(), VARIANT], ids=["t", "variant"])
def test_streaming_frames_match_reference(params, hydrated):
    model, tensors = hydrated(params)
    out_error, state_error = max_error(model, params, tensors)
    assert out_error < TOLERANCE and state_error < TOLERANCE


def test_named_streaming_signature():
    model = build(FastEnhancerParams())
    assert model.name == "fastenhancer"
    assert [t.name for t in model.inputs] == ["spec_in", "state_in_0", "state_in_1"]
    assert [tuple(t.shape) for t in model.inputs] == [(None, 257, 1, 2), (None, 16, 20), (None, 16, 20)]
    assert model.output_names == ["spec_out", "state_out_0", "state_out_1"]
    assert [tuple(t.shape) for t in model.outputs] == [(None, 257, 1, 2), (None, 16, 20), (None, 16, 20)]


def test_carried_state_changes_output(hydrated):
    params = FastEnhancerParams()
    model, _ = hydrated(params)
    spec = np.random.default_rng(2).normal(size=(1, 257, 1, 2)).astype(np.float32)
    zeros = [np.zeros((1, 16, 20), np.float32)] * 2
    carried = [np.full((1, 16, 20), 0.5, np.float32)] * 2
    first = keras.ops.convert_to_numpy(model([spec, *zeros])[0])
    second = keras.ops.convert_to_numpy(model([spec, *carried])[0])
    assert np.abs(first - second).max() > 1e-2
    assert first[:, -1].max() == 0 and first[:, -1].min() == 0


def test_the_mapping_refuses_tensors_of_other_params(hydrated, tmp_path, write_safetensors):
    model, tensors = hydrated(FastEnhancerParams())
    before = [w.copy() for w in model.get_weights()]
    path = tmp_path / "variant.safetensors"
    variant = fastenhancer_mapping(VARIANT, "variant", pinned(path, random_tensors(VARIANT), write_safetensors))
    with pytest.raises(ValueError, match="mapped shape"):
        import_weights(model, variant, path)
    assert all(np.array_equal(a, b) for a, b in zip(before, model.get_weights()))


def test_the_release_mapping_renames_only_the_anonymous_initializers():
    by_path = fastenhancer_mapping(FastEnhancerParams(), "paths", FASTENHANCER_T_ONNX.source)
    renamed = {a.sources: b.sources for a, b in zip(by_path.rows, FASTENHANCER_T_ONNX.rows, strict=True) if a != b}
    assert len(renamed) == 14
    assert all(source[0].startswith(("onnx::MatMul_", "onnx::GRU_")) for source in renamed.values())
    assert all(name.startswith("/") for name in FASTENHANCER_T_ONNX.unused)


RELEASE_FILE = os.environ.get("HELIA_EDGE_FASTENHANCER_ONNX")


@pytest.mark.skipif(not RELEASE_FILE, reason="set HELIA_EDGE_FASTENHANCER_ONNX to the pinned fastenhancer_t.spec.onnx")
def test_the_release_imports_and_matches_the_reference():
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("needs onnx")
    from helia_edge.importers.readers import read_onnx

    params = FastEnhancerParams()
    model = build(params)
    report = import_weights(model, FASTENHANCER_T_ONNX, RELEASE_FILE)
    assert len(report.assignments) == len(model.weights) == len(fastenhancer_weight_shapes(params))
    release = read_onnx(Path(RELEASE_FILE))
    by_path = fastenhancer_mapping(params, "paths", FASTENHANCER_T_ONNX.source)
    tensors = {a.sources[0]: release[b.sources[0]] for a, b in zip(by_path.rows, FASTENHANCER_T_ONNX.rows)}
    out_error, state_error = max_error(model, params, tensors)
    assert out_error < TOLERANCE and state_error < TOLERANCE


def test_fixed_filterbanks_are_frozen_normalized_projections():
    pre, post = linear_filterbanks(64, 16)
    assert pre.shape == (64, 16) and post.shape == (16, 64)
    np.testing.assert_allclose(pre.sum(axis=0), 1.0, rtol=1e-6)
    np.testing.assert_allclose(post.sum(axis=0), 1.0, rtol=1e-6)
    # Values of the onnx-vd-v1.0.0 fixed projections; the later upstream formula differs.
    np.testing.assert_allclose(pre[:5, 0], [0.36670548, 0.29802096, 0.2048894, 0.11175784, 0.01862631], atol=1e-6)
    np.testing.assert_allclose(post[0, :3], [1.0, 0.7619048, 0.52380955], atol=1e-6)
    model = build(FastEnhancerParams())
    assert not model.get_layer("rf_pre_proj").trainable
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(model.get_layer("rf_pre_proj").kernel), pre)


def test_keras_file_roundtrip(tmp_path, hydrated):
    params = VARIANT
    model, _ = hydrated(params)
    path = tmp_path / "fastenhancer.keras"
    model.save(path)
    loaded = keras.saving.load_model(path)
    rng = np.random.default_rng(3)
    inputs = [rng.normal(size=(1, *t.shape[1:])).astype(np.float32) for t in model.inputs]
    for a, b in zip(model(inputs), loaded(inputs)):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))


@pytest.mark.skipif(keras.backend.backend() != "tensorflow", reason="LiteRT export needs the TensorFlow backend")
def test_integer_export_ties_the_named_state_pairs():
    pytest.importorskip("ai_edge_litert")
    from helia_edge.export import ExportSpec, export_model, stream_calibration
    from helia_edge.export.result import state_scales_tied

    model = build(FastEnhancerParams(), batch_size=1)
    frames = (np.random.default_rng(0).normal(size=(16, 257, 1, 2)) * 0.5).astype(np.float32)
    result = export_model(
        model,
        ExportSpec(precision="a8w8", io_dtype="int8", mode="keras"),
        stream_calibration(model, {"spec_in": frames}),
    )
    assert sorted(r.pair for r in result.inputs if r.pair is not None) == [0, 1]
    assert state_scales_tied(result.inputs, result.outputs) is True
