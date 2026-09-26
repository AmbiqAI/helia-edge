"""Explicit target roles and shape-changing training boundaries."""

import keras
import numpy as np
import pytest
from helia_edge.layers.preprocessing import Resizing1D, Resizing2D, RandomCrop1D, AugmentationPipeline


def array(x):
    return keras.ops.convert_to_numpy(x)


@pytest.mark.parametrize("two_dim", [False, True])
@pytest.mark.parametrize("first", [False, True])
def test_resize_target_policy(two_dim, first):
    shape = (2, 6, 8, 1) if two_dim else (2, 8, 1)
    x = np.arange(np.prod(shape), dtype="int64").reshape(shape)
    if first:
        x = np.moveaxis(x, -1, 1)
    cls, args = (Resizing2D, (3, 4)) if two_dim else (Resizing1D, (4,))
    fmt = "channels_first" if first else "channels_last"
    tree = {"signals": {"x": x}, "targets": {"clean": x}, "masks": {"valid": x % 2 == 0}}
    with pytest.raises(ValueError, match="target_interpolation"):
        cls(*args, data_format=fmt)(tree)
    layer = cls(*args, data_format=fmt, target_interpolation="signal")
    loaded = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(layer))
    out = loaded(tree)
    np.testing.assert_allclose(array(out["targets"]["clean"]), array(out["signals"]["x"]))
    assert keras.backend.standardize_dtype(out["targets"]["clean"].dtype) == "float32"
    assert keras.backend.standardize_dtype(out["masks"]["valid"].dtype) == "bool"
    discrete = cls(*args, data_format=fmt, target_interpolation="nearest")(tree)
    assert keras.backend.standardize_dtype(discrete["targets"]["clean"].dtype) == "int64"
    np.testing.assert_array_equal(array(out["masks"]["valid"]), array(discrete["masks"]["valid"]))
    with pytest.raises(ValueError, match="target_interpolation"):
        cls(*args, target_interpolation="unknown")


def test_crop_pipeline_explicit_parameters_and_inference_rng():
    crop = RandomCrop1D(3, seed=42)
    pipeline = AugmentationPipeline([crop])
    x = keras.ops.reshape(keras.ops.arange(16, dtype="float32"), (2, 8, 1))
    params = [{"start": keras.ops.convert_to_tensor([1, 4], dtype="int32")}]
    before = array(crop.generator.state).copy()
    np.testing.assert_array_equal(array(pipeline(x, training=False, transformations=params)), array(x))
    out = pipeline(x, training=True, transformations=params)
    np.testing.assert_array_equal(array(out), np.stack([array(x)[0, 1:4], array(x)[1, 4:7]]))
    np.testing.assert_array_equal(array(crop.generator.state), before)
    with pytest.raises(ValueError, match="one entry"):
        pipeline(x, training=True, transformations=[])


def test_crop_functional_variable_length_and_tensor_training():
    crop = RandomCrop1D(3, seed=42)
    inp = keras.Input((8, 1))
    # A length-independent consumer works in either mode; fixed flatten+dense does not.
    model = keras.Model(inp, keras.layers.GlobalAveragePooling1D()(crop(inp)))
    x = keras.ops.reshape(keras.ops.arange(16, dtype="float32"), (2, 8, 1))
    assert model(x, training=True).shape == model(x, training=False).shape == (2, 1)
    fixed = keras.Model(inp, keras.layers.Dense(1)(keras.layers.Flatten()(crop(inp))))
    assert fixed(x, training=False).shape == (2, 1)
    with pytest.raises((ValueError, RuntimeError)):
        fixed(x, training=True)
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        pipe = AugmentationPipeline([crop])
        fn = tf.function(lambda value, training: pipe(value, training=training))
        assert fn(x, tf.constant(True)).shape == (2, 3, 1)
        state = array(crop.generator.state).copy()
        assert fn(x, tf.constant(False)).shape == (2, 8, 1)
        np.testing.assert_array_equal(array(crop.generator.state), state)
