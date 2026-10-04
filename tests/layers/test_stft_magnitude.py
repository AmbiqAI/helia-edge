"""StftMagnitude: reflect padding, strided frames with a stored basis, and the exact magnitude, on every backend."""

import keras
import numpy as np
import pytest

from helia_edge.layers import StftMagnitude


def reference(x, basis, frame_length, frame_step, bins, padding):
    padded = np.pad(x, ((0, 0), padding), mode="reflect")
    count = (padded.shape[1] - frame_length) // frame_step + 1
    frames = np.stack([padded[:, i * frame_step : i * frame_step + frame_length] for i in range(count)], axis=1)
    spectrum = frames @ basis[:, 0, :]
    return np.sqrt(spectrum[..., :bins] ** 2 + spectrum[..., bins:] ** 2)


@pytest.mark.parametrize("padding", [(0, 0), (0, 8), (5, 3)])
def test_matches_framing_with_the_basis(padding):
    layer = StftMagnitude(frame_length=16, frame_step=8, bins=9, padding=padding)
    x = np.random.default_rng(0).standard_normal((2, 64)).astype(np.float32)
    layer.build(x.shape)
    basis = np.random.default_rng(1).standard_normal((16, 1, 18)).astype(np.float32)
    layer.basis.assign(basis)
    out = keras.ops.convert_to_numpy(layer(x))
    want = reference(x, basis, 16, 8, 9, padding)
    assert out.shape == want.shape == layer.compute_output_shape(x.shape)
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-5)


def test_the_basis_is_a_stored_non_trainable_weight():
    layer = StftMagnitude(frame_length=16, frame_step=8, bins=9, padding=(0, 8))
    layer.build((1, 64))
    assert [w.name for w in layer.weights] == ["basis"] and not layer.trainable_weights
    config = layer.get_config()
    assert config["padding"] == (0, 8)
    assert StftMagnitude.from_config(config).get_config() == config


def test_the_projected_magnitude_is_within_its_bound_without_a_square_root():
    x = np.random.default_rng(2).standard_normal((2, 64)).astype(np.float32)
    basis = np.random.default_rng(3).standard_normal((16, 1, 18)).astype(np.float32)
    exact, projected = (StftMagnitude(16, 8, 9, (0, 8), magnitude=m) for m in ("sqrt", "max_projection"))
    for layer in (exact, projected):
        layer.build(x.shape)
        layer.basis.assign(basis)
    want, got = keras.ops.convert_to_numpy(exact(x)), keras.ops.convert_to_numpy(projected(x))
    assert got.shape == want.shape == projected.compute_output_shape(x.shape)
    relative = np.abs(got - want)[want > 1e-3] / want[want > 1e-3]
    assert relative.max() <= 0.0025
    assert StftMagnitude.from_config(projected.get_config()).magnitude == "max_projection"
    with pytest.raises(ValueError, match="magnitude"):
        StftMagnitude(16, 8, 9, magnitude="power")
