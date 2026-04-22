"""Tests for EmaResidualVectorQuantizer."""

import numpy as np
import pytest

import keras

from helia_edge.layers import EmaResidualVectorQuantizer


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture()
def rvq():
    """Single-level EMA RVQ with small codebook for fast tests."""
    layer = EmaResidualVectorQuantizer(
        num_levels=2,
        num_embeddings=16,
        embedding_dim=8,
        beta=0.25,
        ema_decay=0.99,
    )
    layer.build((None, 8))
    return layer


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #


def test_build_creates_non_trainable_codebooks(rvq):
    """Codebooks, counts, weights should all be non-trainable."""
    assert len(rvq._codebooks) == 2
    assert len(rvq._ema_counts) == 2
    assert len(rvq._ema_weights) == 2
    assert rvq.trainable_weights == []
    assert len(rvq.non_trainable_weights) == 6  # 3 per level * 2 levels


def test_per_level_num_embeddings():
    """Accept per-level codebook sizes."""
    layer = EmaResidualVectorQuantizer(
        num_levels=2,
        num_embeddings=[32, 16],
        embedding_dim=4,
    )
    layer.build((None, 4))
    assert layer.Ks == [32, 16]
    assert layer._codebooks[0].shape == (32, 4)
    assert layer._codebooks[1].shape == (16, 4)


def test_invalid_args():
    """Bad constructor args should raise."""
    with pytest.raises(ValueError):
        EmaResidualVectorQuantizer(num_levels=0, num_embeddings=8, embedding_dim=4)
    with pytest.raises(ValueError):
        EmaResidualVectorQuantizer(num_levels=1, num_embeddings=8, embedding_dim=0)
    with pytest.raises(ValueError):
        EmaResidualVectorQuantizer(num_levels=1, num_embeddings=8, embedding_dim=4, beta=-1)
    with pytest.raises(ValueError):
        EmaResidualVectorQuantizer(num_levels=2, num_embeddings=[8], embedding_dim=4)


# ------------------------------------------------------------------ #
# Forward pass
# ------------------------------------------------------------------ #


def test_output_shape(rvq):
    """Output shape must match input shape."""
    x = np.random.randn(4, 8).astype(np.float32)
    y = rvq(x, training=False)
    assert y.shape == (4, 8)


def test_output_shape_nd(rvq):
    """Works with higher-rank inputs (batch, time, D)."""
    x = np.random.randn(2, 10, 8).astype(np.float32)
    y = rvq(x, training=False)
    assert y.shape == (2, 10, 8)


def test_return_indices(rvq):
    """return_indices=True should give per-level index tensors."""
    x = np.random.randn(4, 8).astype(np.float32)
    y, indices = rvq(x, training=False, return_indices=True)
    assert y.shape == (4, 8)
    assert len(indices) == 2
    for idx in indices:
        assert idx.shape == (4,)


def test_commitment_loss_added(rvq):
    """Layer should add commitment losses (one per level)."""
    x = np.random.randn(4, 8).astype(np.float32)
    _ = rvq(x, training=True)
    assert len(rvq.losses) == 2  # one per level
    for loss in rvq.losses:
        assert float(loss) >= 0.0


# ------------------------------------------------------------------ #
# EMA updates
# ------------------------------------------------------------------ #


def test_ema_updates_codebook():
    """Codebook should change after a training forward pass."""
    layer = EmaResidualVectorQuantizer(
        num_levels=1, num_embeddings=8, embedding_dim=4, ema_decay=0.9,
    )
    layer.build((None, 4))
    cb_before = np.array(layer._codebooks[0])

    x = np.random.randn(32, 4).astype(np.float32)
    _ = layer(x, training=True)

    cb_after = np.array(layer._codebooks[0])
    assert not np.allclose(cb_before, cb_after), "Codebook should change with EMA"


def test_no_ema_at_inference():
    """Codebook should NOT change during inference."""
    layer = EmaResidualVectorQuantizer(
        num_levels=1, num_embeddings=8, embedding_dim=4, ema_decay=0.9,
    )
    layer.build((None, 4))

    # Run one training step to move away from init
    x = np.random.randn(32, 4).astype(np.float32)
    _ = layer(x, training=True)
    cb_before = np.array(layer._codebooks[0])

    # Inference pass should not change codebook
    _ = layer(x, training=False)
    cb_after = np.array(layer._codebooks[0])
    np.testing.assert_array_equal(cb_before, cb_after)


# ------------------------------------------------------------------ #
# Encode / decode round-trip
# ------------------------------------------------------------------ #


def test_encode_decode_matches_forward(rvq):
    """encode() + decode() should produce the same output as forward pass."""
    x = np.random.randn(8, 8).astype(np.float32)
    y, indices_fwd = rvq(x, training=False, return_indices=True)

    indices_enc = rvq.encode(x)
    y_dec = rvq.decode(indices_enc, x.shape)

    # Indices should match
    for i_fwd, i_enc in zip(indices_fwd, indices_enc):
        np.testing.assert_array_equal(np.array(i_fwd), np.array(i_enc))

    # Decoded output should match forward output
    np.testing.assert_allclose(np.array(y), np.array(y_dec), atol=1e-6)


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #


def test_metrics_populated(rvq):
    """Metrics list should contain per-level + aggregate trackers."""
    # 2 levels * 3 per-level + 3 aggregates = 9
    assert len(rvq.metrics) == 9
    names = {m.name for m in rvq.metrics}
    assert "rvq_l1_perplexity" in names
    assert "rvq_l2_usage" in names
    assert "rvq_perplexity_mean" in names
    assert "rvq_usage_mean" in names
    assert "rvq_bits_per_index_sum" in names


# ------------------------------------------------------------------ #
# Serialization
# ------------------------------------------------------------------ #


def test_get_config_roundtrip(rvq):
    """get_config / from_config round-trip."""
    cfg = rvq.get_config()
    restored = EmaResidualVectorQuantizer.from_config(cfg)
    assert restored.M == rvq.M
    assert restored.Ks == rvq.Ks
    assert restored.D == rvq.D
    assert restored.beta == rvq.beta
    assert restored.ema_decay == rvq.ema_decay
    assert restored.epsilon == rvq.epsilon


# ------------------------------------------------------------------ #
# Integration with Model.fit
# ------------------------------------------------------------------ #


def test_model_fit_integration():
    """EMA RVQ should work inside a simple Model.fit loop."""
    enc = keras.layers.Dense(8)
    rvq = EmaResidualVectorQuantizer(
        num_levels=2, num_embeddings=16, embedding_dim=8,
    )
    dec = keras.layers.Dense(4)

    inp = keras.Input((4,))
    z = enc(inp)
    zq = rvq(z)
    out = dec(zq)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    x = np.random.randn(32, 4).astype(np.float32)
    hist = model.fit(x, x, epochs=2, batch_size=16, verbose=0)
    assert "loss" in hist.history
    assert len(hist.history["loss"]) == 2
