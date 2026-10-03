"""The Silero VAD v6 NPU lowering: block STFT and projection magnitude, and the lowered model against the semantic one."""

import keras
import numpy as np
import pytest

from helia_edge.layers import StftMagnitude
from helia_edge.models import lower_silero_vad_v6_npu, silero_vad_v6
from helia_edge.models.silero_vad_npu import SileroNpuFrontend, _block_kernels, _projection_kernel


def randomized(seed=0):
    """A semantic model with random weights scaled so every layer stays in a useful range."""
    keras.utils.set_random_seed(seed)
    model = silero_vad_v6()
    rng = np.random.default_rng(seed)
    for weight in model.weights:
        shape = tuple(weight.shape)
        scale = 0.1 if len(shape) == 1 else 1 / np.sqrt(np.prod(shape[:-1]))
        scale = 4.0 if "basis" in weight.path else scale
        weight.assign((scale * rng.standard_normal(shape)).astype(np.float32))
    return model


def calls(count, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(512 * count + 64) / 16000
    signal = 0.05 * rng.standard_normal(t.size) + 0.4 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    return np.stack([signal[i * 512 : i * 512 + 576] for i in range(count)]).astype(np.float32)


def stream(model, frames):
    h = c = np.zeros((1, 128), np.float32)
    out = []
    for x in frames:
        p, h, c = (keras.ops.convert_to_numpy(v) for v in model([x[None], h, c], training=False))
        out.append((float(p.ravel()[0]), h, c))
    return out


def test_the_frontend_magnitude_is_within_the_projection_bound_of_the_stft():
    rng = np.random.default_rng(2)
    basis = rng.standard_normal((256, 1, 258)).astype(np.float32)
    exact = StftMagnitude(frame_length=256, frame_step=128, bins=129, padding=(0, 64))
    exact.build((1, 576))
    exact.basis.assign(basis)
    frontend = SileroNpuFrontend()
    frontend.build((1, 576))
    frontend.set_weights([*_block_kernels(basis), _projection_kernel()])
    x = calls(3, seed=3)
    want = keras.ops.convert_to_numpy(exact(x))  # (3, 4, 129)
    got = keras.ops.convert_to_numpy(frontend(x))[:, :, 0, :]
    relative = np.abs(got - want) / np.maximum(want, 1e-6)
    assert relative[want > 1e-3].max() <= 0.0025
    # The bound is reached, so the projection is not exact by accident
    assert relative[want > 1e-3].max() > 0.001


def test_the_last_frame_folds_the_reflection():
    rng = np.random.default_rng(4)
    basis = rng.standard_normal((256, 1, 258)).astype(np.float32)
    main, folded = _block_kernels(basis)
    x = rng.standard_normal(576).astype(np.float32)
    padded = np.concatenate([x, x[-2:-66:-1]])
    taps = basis[:, 0, :][:, np.stack([np.arange(129), np.arange(129) + 129], axis=1).reshape(-1)]
    want_last = padded[384:640] @ taps
    np.testing.assert_allclose(x[384:576] @ folded.reshape(192, 258), want_last, rtol=1e-4, atol=1e-3)
    want_first = padded[0:256] @ taps
    np.testing.assert_allclose(x[0:256] @ main.reshape(256, 258), want_first, rtol=1e-4, atol=1e-3)


def test_the_lowered_model_tracks_the_semantic_model():
    semantic = randomized()
    lowered = lower_silero_vad_v6_npu(semantic)
    assert [t.name for t in lowered.inputs] == [t.name for t in semantic.inputs]
    assert lowered.output_names == semantic.output_names
    frames = calls(24)
    got, want = stream(lowered, frames), stream(semantic, frames)
    probs = np.array([p for p, _, _ in want])
    assert probs.std() > 1e-3
    for (p, h, c), (rp, rh, rc) in zip(got, want, strict=True):
        assert abs(p - rp) < 0.002
        np.testing.assert_allclose(h, rh, atol=0.02)
        np.testing.assert_allclose(c, rc, atol=0.05)


def test_the_lowering_copies_weights_and_leaves_the_semantic_model_alone():
    semantic = randomized(5)
    before = [keras.ops.convert_to_numpy(w) for w in semantic.weights]
    lowered = lower_silero_vad_v6_npu(semantic)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(lowered.get_layer("lstm").kernel),
        keras.ops.convert_to_numpy(semantic.get_layer("lstm").kernel),
    )
    k3 = keras.ops.convert_to_numpy(semantic.get_layer("encoder2").kernel)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(lowered.get_layer("conv3").kernel), np.concatenate([k3[1], k3[2]])
    )
    for got, want in zip((keras.ops.convert_to_numpy(w) for w in semantic.weights), before, strict=True):
        np.testing.assert_array_equal(got, want)


def test_the_lowered_model_has_no_square_root_or_strided_stft():
    if keras.backend.backend() != "tensorflow":
        pytest.skip("LiteRT export runs on the TensorFlow backend")
    from helia_edge.export import ExportSpec, export_model
    from helia_edge.export.litert import operator_names

    lowered_ops = set(
        operator_names(
            export_model(
                randomized(),
                ExportSpec(precision="fp32", io_dtype="float32", mode="keras", lowering="npu"),
                architecture="vad_silero_v6",
            ).content
        )
    )
    assert "SQRT" not in lowered_ops and "MIRROR_PAD" not in lowered_ops
    assert {"ABS", "REDUCE_MAX", "CONV_2D"} <= lowered_ops
