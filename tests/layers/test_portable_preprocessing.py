"""Numerical and structured-input contracts for the first portable transforms."""

import importlib.util
import sys

import keras
import numpy as np
import pytest

from helia_edge.layers.preprocessing import FirFilter, Normalization1D, RandomGaussianNoise1D


def array(value):
    return keras.ops.convert_to_numpy(value)


def payload(batched=True):
    shape = (2, 8, 2) if batched else (8, 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return {
        "data": keras.ops.convert_to_tensor(x),
        "targets": {"seg": keras.ops.convert_to_tensor(x, dtype="int32")},
        "masks": {"valid": keras.ops.convert_to_tensor(x % 2 == 0)},
    }


def test_torch_import_without_tensorflow():
    if keras.backend.backend() != "torch":
        pytest.skip("Torch-only environment contract")
    assert importlib.util.find_spec("tensorflow") is None
    assert "tensorflow" not in sys.modules


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("channels_first", [True, False])
def test_normalization_reference(batched, channels_first):
    x = array(payload(batched)["data"])
    expected = (x - [1, 2]) / np.sqrt(np.array([4, 9]) + 1e-6)
    if channels_first:
        x = np.swapaxes(x, -1, -2)
        expected = np.swapaxes(expected, -1, -2)
    layer = Normalization1D([1, 2], [4, 9], data_format="channels_first" if channels_first else "channels_last")
    np.testing.assert_allclose(array(layer(x)), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batched", [True, False])
def test_legacy_preserves_auxiliary_dtype_and_input(batched):
    inp = payload(batched)
    original = inp["data"]
    before = array(original).copy()
    out = Normalization1D(1.0, 4.0)(inp)
    assert inp["data"] is original
    np.testing.assert_array_equal(array(original), before)
    assert keras.backend.standardize_dtype(out["targets"]["seg"].dtype) == "int32"
    assert keras.backend.standardize_dtype(out["masks"]["valid"].dtype) == "bool"
    np.testing.assert_array_equal(array(out["targets"]["seg"]), before.astype("int32"))


def test_sample_conversion_identity_and_new_containers():
    from helia_edge.layers.preprocessing import Sample

    inp = payload()
    sample = Sample(signals={"ecg": inp["data"]}, targets=inp["targets"], masks=inp["masks"])
    tree = sample.tensor_tree()
    assert tree["signals"] is not sample.signals and tree["signals"]["ecg"] is inp["data"]
    assert tree["targets"] is not sample.targets and tree["targets"]["seg"] is inp["targets"]["seg"]
    out = Normalization1D(1.0, 4.0)(tree)
    np.testing.assert_allclose(array(out["signals"]["ecg"]), (array(inp["data"]) - 1) / np.sqrt(4.000001), atol=1e-6)
    np.testing.assert_array_equal(array(out["masks"]["valid"]), array(inp["masks"]["valid"]))
    assert tree["signals"]["ecg"] is inp["data"]


@pytest.mark.parametrize("forward_backward", [False, True])
@pytest.mark.parametrize("channels_first", [False, True])
def test_fir_reference_multichannel(forward_backward, channels_first):
    x = np.arange(32, dtype=np.float32).reshape(2, 8, 2)
    taps = np.array([0.1, 0.2, 0.7], dtype=np.float32)

    # depthwise_conv uses cross-correlation, retaining the existing FIR convention.
    def reference(v):
        return np.stack(
            [np.stack([np.convolve(row[:, c], taps[::-1], mode="same") for c in range(2)], axis=-1) for row in v]
        )

    expected = reference(x)
    if forward_backward:
        expected = reference(expected[:, ::-1, :])[:, ::-1, :]
    if channels_first:
        x = x.transpose(0, 2, 1)
        expected = expected.transpose(0, 2, 1)
    layer = FirFilter(
        taps, forward_backward=forward_backward, data_format="channels_first" if channels_first else "channels_last"
    )
    np.testing.assert_allclose(array(layer(x)), expected, atol=2e-6, rtol=1e-6)


def test_fir_config_roundtrip():
    layer = FirFilter(np.array([0.1, 0.2, 0.7], dtype=np.float32), forward_backward=True)
    restored = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(layer))
    x = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
    np.testing.assert_array_equal(array(restored(x)), array(layer(x)))


@pytest.mark.parametrize("kind", ["normalization", "fir", "noise"])
def test_nested_functional_serialization(tmp_path, kind):
    from helia_edge.layers.preprocessing import Sample

    inp = payload()
    tree = Sample({"ecg": inp["data"]}, inp["targets"], inp["masks"]).tensor_tree()
    symbolic = keras.tree.map_structure(
        lambda v: keras.Input(shape=v.shape[1:], dtype=keras.backend.standardize_dtype(v.dtype)), tree
    )
    layer = {
        "normalization": lambda: Normalization1D(1.0, 4.0),
        "fir": lambda: FirFilter(np.array([0.25, 0.5, 0.25])),
        "noise": lambda: RandomGaussianNoise1D(0.1, seed=42),
    }[kind]()
    model = keras.Model(symbolic, layer(symbolic, training=False))
    model.save(tmp_path / f"{kind}.keras")
    restored = keras.models.load_model(tmp_path / f"{kind}.keras")
    result = restored(tree)
    for actual, expected in zip(keras.tree.flatten(result), keras.tree.flatten(model(tree)), strict=True):
        np.testing.assert_allclose(array(actual), array(expected), atol=1e-6, rtol=1e-6)
    assert keras.backend.standardize_dtype(result["targets"]["seg"].dtype) == "int32"
    assert keras.backend.standardize_dtype(result["masks"]["valid"].dtype) == "bool"


def test_noise_rng_inference_and_targets():
    a = RandomGaussianNoise1D((0.1, 0.1), seed=42)
    b = RandomGaussianNoise1D((0.1, 0.1), seed=42)
    inp = payload()
    np.testing.assert_array_equal(array(a(inp, training=False)["data"]), array(inp["data"]))
    first = a(inp, training=True)
    np.testing.assert_array_equal(array(first["data"]), array(b(inp, training=True)["data"]))
    assert not np.array_equal(array(first["data"]), array(a(inp, training=True)["data"]))
    np.testing.assert_array_equal(array(first["targets"]["seg"]), array(inp["targets"]["seg"]))
    np.testing.assert_array_equal(array(first["masks"]["valid"]), array(inp["masks"]["valid"]))


def test_tf_data_preserves_integer_targets_and_boolean_masks():
    if keras.backend.backend() != "tensorflow":
        pytest.skip("TensorFlow-backend integration")
    import tensorflow as tf

    inp = payload()
    out = next(iter(tf.data.Dataset.from_tensors(inp).map(Normalization1D(1.0, 4.0))))
    assert out["targets"]["seg"].dtype == tf.int32
    assert out["masks"]["valid"].dtype == tf.bool
    np.testing.assert_array_equal(out["targets"]["seg"], inp["targets"]["seg"])


@pytest.mark.parametrize("kind", ["normalization", "fir", "noise"])
def test_compiled(kind):
    layer = {
        "normalization": lambda: Normalization1D(1.0, 4.0),
        "fir": lambda: FirFilter(np.array([0.25, 0.5, 0.25])),
        "noise": lambda: RandomGaussianNoise1D(0.1, seed=42),
    }[kind]()
    x = payload()["data"]
    layer(x, training=False)
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        fn = tf.function(layer)
    else:
        import torch

        fn = torch.compile(layer, backend="eager", fullgraph=kind != "noise")
    out = fn(x, training=False)
    np.testing.assert_allclose(array(out), array(layer(x, training=False)), atol=1e-6, rtol=1e-6)
    if kind == "noise":
        assert not np.array_equal(array(fn(x, training=True)), array(x))


def test_tensor_training_flag():
    if keras.backend.backend() != "tensorflow":
        pytest.skip("TensorFlow dynamic training flag")
    import tensorflow as tf

    layer = RandomGaussianNoise1D(0.1, seed=42)
    fn = tf.function(layer)
    x = payload()["data"]
    np.testing.assert_array_equal(array(fn(x, training=tf.constant(False))), array(x))
    assert not np.array_equal(array(fn(x, training=tf.constant(True))), array(x))


def test_invalid_rank_and_ambiguous_schema():
    with pytest.raises(ValueError, match="rank 2 or 3"):
        Normalization1D(0.0, 1.0)(keras.ops.zeros((3,)))
    with pytest.raises(ValueError, match="either signals"):
        Normalization1D(0.0, 1.0)({"data": keras.ops.zeros((8, 1)), "signals": {"x": keras.ops.zeros((8, 1))}})


def test_legacy_2d_import_and_behavior():
    from helia_edge.layers.preprocessing.normalization import Normalization2D

    x = keras.ops.ones((1, 2, 2, 1))
    np.testing.assert_allclose(array(Normalization2D(1.0, 4.0)(x)), 0.0, atol=1e-6)
