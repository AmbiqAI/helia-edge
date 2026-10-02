"""Silero VAD v6: the mapping covers the model, the model matches a NumPy transcription of the reference
graph, and (with the pinned ONNX file) ONNX Runtime."""

import hashlib
import importlib.util
import os

import keras
import numpy as np
import pytest

from helia_edge.importers import SourcePin, import_weights
from helia_edge.models.silero_vad import SAMPLES, SILERO_VAD_V6_ONNX, UNITS, silero_vad_v6

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


@pytest.fixture
def imported(tmp_path, write_safetensors):
    """The model with synthetic weights imported through the Silero mapping, from a safetensors file."""
    tensors = synthetic_tensors()
    path = tmp_path / "silero.safetensors"
    write_safetensors(path, tensors)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    mapping = SILERO_VAD_V6_ONNX.model_copy(
        update={"format": "safetensors", "source": SourcePin(uri="file://s", sha256=sha256)}
    )
    model = silero_vad_v6()
    report = import_weights(model, mapping, path)
    return model, tensors, report


def test_the_model_has_the_streaming_interface():
    model = silero_vad_v6()
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


def test_the_model_saves_and_reloads(imported, tmp_path):
    model, _, _ = imported
    model.save(tmp_path / "silero.keras")
    loaded = keras.saving.load_model(tmp_path / "silero.keras")
    state = np.random.default_rng(3).standard_normal((2, 1, UNITS)).astype(np.float32)
    feed = [audio(2)[None, :SAMPLES], state[0], state[1]]
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


@pytest.mark.skipif(
    not (ONNX_FILE and AUDIO_FILE),
    reason="set HELIA_EDGE_SILERO_ONNX to the pinned silero_vad_16k_op15.onnx and HELIA_EDGE_SILERO_AUDIO to 16 kHz speech",
)
def test_imported_weights_match_onnx_runtime():
    """A1: on speech, probability within 1e-4 of ONNX Runtime and the same decisions at 0.5, state carried."""
    if importlib.util.find_spec("onnxruntime") is None or importlib.util.find_spec("onnx") is None:
        pytest.skip("needs onnx and onnxruntime")
    import onnxruntime

    model = silero_vad_v6()
    import_weights(model, SILERO_VAD_V6_ONNX, ONNX_FILE)
    session = onnxruntime.InferenceSession(ONNX_FILE)
    state = np.zeros((2, 1, UNITS), np.float32)

    def onnx_step(x, h, c):
        nonlocal state
        prob, state = session.run(None, {"input": x[None], "state": state, "sr": np.array(16000, np.int64)})
        return prob, state[0, 0], state[1, 0]

    signal = speech()
    got, want = stream(keras_step(model), signal), stream(onnx_step, signal)
    probs, wanted = np.array([p for p, _, _ in got]), np.array([p for p, _, _ in want])
    assert np.abs(probs - wanted).max() <= 1e-4
    assert ((probs > 0.5) == (wanted > 0.5)).all()
    assert (wanted > 0.5).any() and (wanted < 0.5).any()
