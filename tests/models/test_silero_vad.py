"""Silero VAD v6: the mapping covers the model, the model matches a NumPy transcription of the reference
graph, and (with the pinned ONNX file) ONNX Runtime."""

import hashlib
import importlib.util
import itertools
import os

import keras
import numpy as np
import pytest

from helia_edge.importers import SourcePin, import_weights
from helia_edge.layers import StftMagnitude
from helia_edge.models import SileroVadParams
from helia_edge.models.silero_vad import SAMPLES, UNITS, SileroBlockStft, SileroLiveTaps, build
from helia_edge.models.silero_vad_params import SILERO_VAD_V6_ONNX

SHAPES = {
    "model.stft.forward_basis_buffer": (258, 1, 256),
    "model.encoder.0.reparam_conv.weight": (128, 129, 3),
    "model.encoder.0.reparam_conv.bias": (128,),
    "model.encoder.1.reparam_conv.weight": (64, 128, 3),
    "model.encoder.1.reparam_conv.bias": (64,),
    "model.encoder.2.reparam_conv.weight": (64, 64, 3),
    "model.encoder.2.reparam_conv.bias": (64,),
    "model.encoder.3.reparam_conv.weight": (128, 64, 3),
    "model.encoder.3.reparam_conv.bias": (128,),
    "model.decoder.rnn.weight_ih": (512, 128),
    "model.decoder.rnn.weight_hh": (512, 128),
    "model.decoder.rnn.bias_ih": (512,),
    "model.decoder.rnn.bias_hh": (512,),
    "model.decoder.decoder.2.weight": (1, 128, 1),
    "model.decoder.decoder.2.bias": (1,),
}
"""Initializer names and shapes of silero_vad_16k_op15.onnx (v6.2.2)."""


def synthetic_tensors(seed=0):
    rng = np.random.default_rng(seed)
    tensors = {
        name: (rng.standard_normal(shape) / np.sqrt(np.prod(shape[1:]) or 1)).astype(np.float32)
        for name, shape in SHAPES.items()
    }
    tensors["model.stft.forward_basis_buffer"] *= 4
    return tensors


def reference_step(w, x, h, c):
    """NumPy transcription of the v6.2.2 16 kHz graph for one call: x [576], h and c [128]."""
    padded = np.concatenate([x, x[-2:-66:-1]])  # reflect 64 samples on the right
    frames = np.stack([padded[i * 128 : i * 128 + 256] for i in range(4)])
    spectrum = frames @ w["model.stft.forward_basis_buffer"][:, 0, :].T
    y = np.sqrt(spectrum[:, :129] ** 2 + spectrum[:, 129:] ** 2).T  # [129, 4]
    for index, stride in enumerate((1, 2, 2, 1)):
        weight, bias = w[f"model.encoder.{index}.reparam_conv.weight"], w[f"model.encoder.{index}.reparam_conv.bias"]
        y = np.pad(y, ((0, 0), (1, 1)))
        columns = np.stack([y[:, t * stride : t * stride + 3] for t in range((y.shape[1] - 3) // stride + 1)])
        y = np.maximum(np.einsum("tck,ock->ot", columns, weight) + bias[:, None], 0)
    z = w["model.decoder.rnn.weight_ih"] @ y[:, 0] + w["model.decoder.rnn.bias_ih"]
    z = z + w["model.decoder.rnn.weight_hh"] @ h + w["model.decoder.rnn.bias_hh"]
    i, f, g, o = np.split(z, 4)
    sigmoid = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
    h = sigmoid(o) * np.tanh(c)
    logit = w["model.decoder.decoder.2.weight"][0, :, 0] @ np.maximum(h, 0) + w["model.decoder.decoder.2.bias"][0]
    return sigmoid(logit), h, c


def audio(calls, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(512 * calls) / 16000
    signal = 0.05 * rng.standard_normal(t.size) + 0.4 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    return signal.astype(np.float32)


def stream(step, signal):
    """Call ``step(x, h, c)`` per 512 new samples with 64 samples of context; return prob, h, c per call."""
    context, h, c = np.zeros(64, np.float32), np.zeros(UNITS, np.float32), np.zeros(UNITS, np.float32)
    out = []
    for start in range(0, signal.size, 512):
        x = np.concatenate([context, signal[start : start + 512]])
        prob, h, c = step(x, h, c)
        out.append((float(np.ravel(prob)[0]), h, c))
        context = x[-64:]
    return out


def keras_step(model):
    def step(x, h, c):
        prob, h, c = model([x[None], h[None], c[None]], training=False)
        return (keras.ops.convert_to_numpy(v)[0] for v in (prob, h, c))

    return step


VARIANTS = [
    SileroVadParams(stft=stft, magnitude=magnitude, encoder_tail=tail)
    for stft, magnitude, tail in itertools.product(
        ("conv1d", "conv_blocks"), ("sqrt", "max_projection"), ("conv", "live_taps")
    )
]
VARIANT_IDS = ["/".join((p.stft, p.magnitude, p.encoder_tail)) for p in VARIANTS]
NPU = SileroVadParams(stft="conv_blocks", magnitude="max_projection", encoder_tail="live_taps")
"""The options that run on integer NPUs: no square root, no reflect padding, no strided frames over samples."""


def frontend(stft, magnitude="sqrt"):
    """The model's ``stft`` layer for these options, built for one call."""
    layer = SileroBlockStft(magnitude) if stft == "conv_blocks" else StftMagnitude(256, 128, 129, (0, 64), magnitude)
    layer.build((1, SAMPLES))
    return layer


@pytest.fixture
def importer(tmp_path, write_safetensors):
    """Build ``params`` and import synthetic weights through the Silero mapping, from a safetensors file."""
    tensors = synthetic_tensors()
    path = tmp_path / "silero.safetensors"
    write_safetensors(path, tensors)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    mapping = SILERO_VAD_V6_ONNX.model_copy(
        update={"source": SourcePin(uri="file://s", sha256=sha256, format="safetensors")}
    )

    def load(params=SileroVadParams()):
        model = build(params, batch_size=1)
        return model, tensors, import_weights(model, mapping, path)

    return load


@pytest.fixture
def imported(importer):
    """The default model with synthetic weights imported through the Silero mapping."""
    return importer()


def test_the_model_has_the_streaming_interface():
    model = build(SileroVadParams(), batch_size=1)
    assert [t.name for t in model.inputs] == ["audio", "state_in_0", "state_in_1"]
    assert model.output_names == ["prob", "state_out_0", "state_out_1"]
    assert tuple(model.inputs[0].shape) == (1, SAMPLES)


def test_the_mapping_uses_every_initializer_and_sets_every_weight(imported):
    model, tensors, report = imported
    used = {name for _, sources in report.assignments for name in sources}
    assert used == set(SHAPES) == set(tensors)
    assert sorted(path for path, _ in report.assignments) == sorted(w.path for w in model.weights)


def test_the_model_matches_the_reference_graph(imported):
    model, tensors, _ = imported
    signal = audio(24)
    got = stream(keras_step(model), signal)
    want = stream(lambda x, h, c: reference_step(tensors, x, h, c), signal)
    probs = np.array([p for p, _, _ in want])
    assert probs.std() > 1e-3
    for (p, h, c), (rp, rh, rc) in zip(got, want, strict=True):
        assert abs(p - rp) < 1e-5
        np.testing.assert_allclose(h, rh, atol=1e-5)
        np.testing.assert_allclose(c, rc, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("params", VARIANTS, ids=VARIANT_IDS)
def test_every_variant_takes_the_same_weights(importer, params):
    model, tensors, report = importer(params)
    default = build(SileroVadParams(), batch_size=1)
    assert [t.name for t in model.inputs] == ["audio", "state_in_0", "state_in_1"]
    assert model.output_names == ["prob", "state_out_0", "state_out_1"]
    assert {w.path: tuple(w.shape) for w in model.weights} == {w.path: tuple(w.shape) for w in default.weights}
    assert {name for _, sources in report.assignments for name in sources} == set(tensors)


@pytest.mark.parametrize(
    "params", [p for p in VARIANTS if p.magnitude == "sqrt"], ids=lambda p: f"{p.stft}/{p.encoder_tail}"
)
def test_the_exact_options_match_the_reference_graph(importer, params):
    model, tensors, _ = importer(params)
    signal = audio(24)
    got = stream(keras_step(model), signal)
    want = stream(lambda x, h, c: reference_step(tensors, x, h, c), signal)
    for (p, h, c), (rp, rh, rc) in zip(got, want, strict=True):
        assert abs(p - rp) < 1e-5
        np.testing.assert_allclose(h, rh, atol=1e-5)
        np.testing.assert_allclose(c, rc, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize(
    "params", [p for p in VARIANTS if p.magnitude == "max_projection"], ids=lambda p: f"{p.stft}/{p.encoder_tail}"
)
def test_the_projected_magnitude_tracks_the_reference_graph(importer, params):
    model, tensors, _ = importer(params)
    signal = audio(24)
    got = stream(keras_step(model), signal)
    want = stream(lambda x, h, c: reference_step(tensors, x, h, c), signal)
    assert np.array([p for p, _, _ in want]).std() > 1e-3
    for (p, h, c), (rp, rh, rc) in zip(got, want, strict=True):
        assert abs(p - rp) < 0.002
        np.testing.assert_allclose(h, rh, atol=0.02)
        np.testing.assert_allclose(c, rc, atol=0.05)


@pytest.mark.parametrize("stft", ["conv1d", "conv_blocks"])
def test_the_projected_magnitude_is_within_its_bound(stft):
    basis = synthetic_tensors()["model.stft.forward_basis_buffer"].transpose(2, 1, 0)  # ONNX (258, 1, 256)
    x = np.stack([audio(2, seed=s)[:SAMPLES] for s in range(3)])
    layers = [frontend(stft, magnitude) for magnitude in ("sqrt", "max_projection")]
    for layer in layers:
        layer.basis.assign(basis)
    exact, projected = (keras.ops.convert_to_numpy(layer(x)).reshape(3, 4, 129) for layer in layers)
    relative = np.abs(projected - exact)[exact > 1e-3] / exact[exact > 1e-3]
    assert relative.max() <= 0.0025
    assert relative.max() > 0.001  # the bound is reached, so the projection is not exact by accident


def test_the_block_frames_fold_the_reflection():
    """Block convolutions with the folded last frame give the strided, reflect-padded frames."""
    basis = np.random.default_rng(4).standard_normal((256, 1, 258)).astype(np.float32)
    x = np.random.default_rng(5).standard_normal((2, SAMPLES)).astype(np.float32)
    layers = [frontend(stft) for stft in ("conv1d", "conv_blocks")]
    for layer in layers:
        layer.basis.assign(basis)
    strided, blocks = (keras.ops.convert_to_numpy(layer(x)).reshape(2, 4, 129) for layer in layers)
    np.testing.assert_allclose(blocks, strided, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(keras.backend.backend() != "tensorflow", reason="LiteRT export runs on the TensorFlow backend")
def test_the_npu_options_export_with_folded_weights_and_no_square_root(importer):
    pytest.importorskip("ai_edge_litert")
    import tensorflow as tf

    from helia_edge.export import ExportSpec, export_model
    from helia_edge.export.litert import operator_names

    model, _, _ = importer(NPU)
    content = export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras")).content
    ops = operator_names(content)
    assert not {"SQRT", "MIRROR_PAD", "TRANSPOSE", "GATHER"} & set(ops)
    assert {"ABS", "REDUCE_MAX", "CONV_2D"} <= set(ops)
    # The block kernels, folded reflection and tap slices are derived from the weights in the graph;
    # the converter must fold them, so every filter is a constant
    interpreter = tf.lite.Interpreter(model_content=content)
    produced = {i for op in interpreter._get_ops_details() for i in op["outputs"]}
    filters = [
        op["inputs"][1] for op in interpreter._get_ops_details() if op["op_name"] in ("CONV_2D", "FULLY_CONNECTED")
    ]
    assert filters and not produced & set(filters)


def test_the_default_keeps_its_layers_so_saved_weights_still_load():
    """Keras weight files are keyed by layer class: the default model keeps the classes of earlier versions."""
    model = build(SileroVadParams(), batch_size=1)
    assert type(model.get_layer("stft")) is StftMagnitude
    assert all(type(model.get_layer(f"encoder{i}")) is keras.layers.Conv1D for i in range(4))


def test_the_layers_refuse_other_geometries():
    with pytest.raises(ValueError, match="576 samples"):
        SileroBlockStft().build((1, 2 * SAMPLES))
    for taps in ((), (1, 1), (-1,), (3,)):
        with pytest.raises(ValueError, match="distinct kernel taps"):
            SileroLiveTaps(8, taps)
    with pytest.raises(ValueError, match="split over the taps"):
        SileroLiveTaps(8, (1, 2)).build((1, 127))
    with pytest.raises(ValueError, match="magnitude"):
        SileroBlockStft("power")
    built = SileroBlockStft()
    built.build((1, SAMPLES))
    with pytest.raises(ValueError, match="576 samples"):
        built(np.zeros((1, 2 * SAMPLES), np.float32))


@pytest.mark.parametrize("params", [NPU, SileroVadParams(magnitude="max_projection")], ids=["npu", "conv1d_projection"])
@pytest.mark.parametrize("policy", ["mixed_float16", "mixed_bfloat16"])
def test_the_projected_magnitude_runs_under_mixed_precision(params, policy):
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy(policy)
    try:
        model = build(params, batch_size=1)
        prob, *_ = model([audio(2)[None, :SAMPLES], np.zeros((1, UNITS), np.float32), np.zeros((1, UNITS), np.float32)])
    finally:
        keras.config.set_dtype_policy(previous)
    assert prob.shape == (1, 1)


@pytest.mark.parametrize("params", [SileroVadParams(), NPU], ids=["default", "npu"])
def test_saved_models_load_in_a_new_process(importer, params, tmp_path):
    import subprocess
    import sys

    model, _, _ = importer(params)
    model.save(tmp_path / "silero.keras")
    feed = [audio(2)[None, :SAMPLES], np.zeros((1, UNITS), np.float32), np.zeros((1, UNITS), np.float32)]
    np.save(tmp_path / "want.npy", keras.ops.convert_to_numpy(model(feed)[0]))
    np.save(tmp_path / "audio.npy", feed[0])
    source = f"""
import keras
import numpy as np
from helia_edge.models import load_model
model = load_model({str(tmp_path / "silero.keras")!r})
state = np.zeros((1, {UNITS}), np.float32)
got = keras.ops.convert_to_numpy(model([np.load({str(tmp_path / "audio.npy")!r}), state, state])[0])
np.testing.assert_array_equal(got, np.load({str(tmp_path / "want.npy")!r}))
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_model_saves_and_reloads(imported, tmp_path):
    model, _, _ = imported
    model.save(tmp_path / "silero.keras")
    loaded = keras.saving.load_model(tmp_path / "silero.keras")
    state = np.random.default_rng(3).standard_normal((2, 1, UNITS)).astype(np.float32)
    feed = [audio(2)[None, :SAMPLES], state[0], state[1]]
    for got, want in zip(loaded(feed), model(feed), strict=True):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(got), keras.ops.convert_to_numpy(want))


def test_the_npu_options_save_and_reload(importer, tmp_path):
    model, _, _ = importer(NPU)
    model.save(tmp_path / "silero_npu.keras")
    loaded = keras.saving.load_model(tmp_path / "silero_npu.keras")
    feed = [audio(2)[None, :SAMPLES], *np.random.default_rng(3).standard_normal((2, 1, UNITS)).astype(np.float32)]
    for got, want in zip(loaded(feed), model(feed), strict=True):
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(got), keras.ops.convert_to_numpy(want))


ONNX_FILE = os.environ.get("HELIA_EDGE_SILERO_ONNX")
AUDIO_FILE = os.environ.get("HELIA_EDGE_SILERO_AUDIO")


def speech():
    """16-bit mono 16 kHz speech from HELIA_EDGE_SILERO_AUDIO, in whole calls of 512 samples."""
    import wave

    with wave.open(AUDIO_FILE) as file:
        assert (file.getframerate(), file.getnchannels(), file.getsampwidth()) == (16000, 1, 2)
        samples = np.frombuffer(file.readframes(file.getnframes()), "<i2").astype(np.float32) / 32768
    return samples[: samples.size // 512 * 512]


@pytest.mark.skipif(not ONNX_FILE, reason="set HELIA_EDGE_SILERO_ONNX to the pinned silero_vad_16k_op15.onnx")
@pytest.mark.parametrize(
    ("params", "tolerance"),
    [(SileroVadParams(), (1e-4, 1e-4, 1e-3)), (NPU, (2e-3, 0.02, 0.05))],
    ids=["exact", "npu"],
)
def test_imported_weights_match_onnx_runtime(params, tolerance):
    """A1: probability, h and c within (1e-4, 1e-4, 1e-3) of ONNX Runtime with the state carried, or
    (2e-3, 0.02, 0.05) with the projected magnitude; on speech from HELIA_EDGE_SILERO_AUDIO, also the same
    decisions at 0.5, with speech and non-speech present."""
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("onnx") is None:
        pytest.skip("needs onnx and onnxruntime")
    import onnxruntime

    model = build(params, batch_size=1)
    import_weights(model, SILERO_VAD_V6_ONNX, ONNX_FILE)
    session = onnxruntime.InferenceSession(ONNX_FILE)
    state = np.zeros((2, 1, UNITS), np.float32)

    def onnx_step(x, h, c):
        nonlocal state
        prob, state = session.run(None, {"input": x[None], "state": state, "sr": np.array(16000, np.int64)})
        return prob, state[0, 0], state[1, 0]

    signal = speech() if AUDIO_FILE else audio(160, seed=2)
    got, want = stream(keras_step(model), signal), stream(onnx_step, signal)
    probs, wanted = np.array([p for p, _, _ in got]), np.array([p for p, _, _ in want])
    prob_tolerance, h_tolerance, c_tolerance = tolerance
    assert np.abs(probs - wanted).max() <= prob_tolerance
    for (_, h, c), (_, rh, rc) in zip(got, want, strict=True):
        np.testing.assert_allclose(h, rh, atol=h_tolerance)
        np.testing.assert_allclose(c, rc, atol=c_tolerance, rtol=1e-5)
    if AUDIO_FILE:
        assert ((probs > 0.5) == (wanted > 0.5)).all()
        assert (wanted > 0.5).any() and (wanted < 0.5).any()
