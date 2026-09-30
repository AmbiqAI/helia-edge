"""Explicit target roles and shape-changing training boundaries."""

import keras
import numpy as np
import pytest

from helia_edge.layers.preprocessing import AugmentationPipeline, RandomCrop1D, Resizing1D, Resizing2D


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


@pytest.mark.parametrize("two_dim,batched", [(False, False), (False, True), (True, False), (True, True)])
def test_random_composite_signal_dtype(two_dim, batched):
    from helia_edge.layers import preprocessing as pp

    shape = (2, 4, 6, 1) if two_dim else (2, 6, 1)
    if not batched:
        shape = shape[1:]
    x = keras.ops.ones(shape, dtype="int32")
    child = pp.Rescaling2D if two_dim else pp.Rescaling1D
    pipeline = pp.RandomAugmentation2DPipeline if two_dim else pp.RandomAugmentation1DPipeline
    tree = {
        "signals": {"x": x},
        "targets": {"id": keras.ops.full(shape, 2**40 + 1, dtype="int64")},
        "masks": {"valid": keras.ops.ones(shape, dtype="bool")},
        "extra": keras.ops.convert_to_tensor([7], dtype="int32"),
    }
    layers = [pp.RandomChoice([child(2.0)], seed=42)] + [
        pipeline([child(2.0)], rate=rate, augmentations_per_sample=count, seed=42)
        for rate, count in [(0.0, 1), (0.5, 1), (1.0, 0), (1.0, 1)]
    ]

    def check(out):
        assert keras.backend.standardize_dtype(out["signals"]["x"].dtype) == "float32"
        assert out["signals"]["x"].shape == x.shape
        for group, key in [("targets", "id"), ("masks", "valid")]:
            assert out[group][key].dtype == tree[group][key].dtype
            np.testing.assert_array_equal(array(out[group][key]), array(tree[group][key]))
        np.testing.assert_array_equal(array(out["extra"]), [7])
        assert keras.backend.standardize_dtype(out["extra"].dtype) == "int32"

    for layer in layers:
        before = array(layer.generator.state).copy()
        check(layer(tree, training=False))
        np.testing.assert_array_equal(array(layer.generator.state), before)
        check(layer(tree, training=True))
        for payload in (x, {"data": x, "targets": tree["targets"], "extra": tree["extra"]}):
            result = layer(payload, training=False)
            value = result["data"] if isinstance(result, dict) else result
            assert keras.backend.standardize_dtype(value.dtype) == "float32"
            assert value.shape == x.shape
        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            fn = tf.function(lambda payload, flag: layer(payload, training=flag))
            check(fn(tree, tf.constant(True)))
            before = array(layer.generator.state).copy()
            check(fn(tree, tf.constant(False)))
            np.testing.assert_array_equal(array(layer.generator.state), before)


@pytest.mark.parametrize("two_dim", [False, True])
def test_zero_normal_cutout_does_not_sample(two_dim, monkeypatch):
    from helia_edge.layers import preprocessing as pp

    cls = pp.RandomCutout2D if two_dim else pp.RandomCutout1D
    shape = (2, 4, 6, 1) if two_dim else (2, 6, 1)
    layer = cls(cutouts=0, fill_mode="normal", seed=7)
    x = keras.ops.ones(shape)
    before = array(layer.generator.state).copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled cutout must not allocate random fill")

    monkeypatch.setattr(keras.random, "normal", forbidden)
    np.testing.assert_array_equal(array(layer(x, training=True)), array(x))
    np.testing.assert_array_equal(array(layer.generator.state), before)


@pytest.mark.parametrize("training", [False, True, None])
def test_mixed_schema_rejected_in_all_modes(training):
    from helia_edge.layers import preprocessing as pp

    x = keras.ops.ones((2, 6, 1))
    tree = {"signals": {"x": x}, "data": x}
    for layer in [pp.RandomCrop1D(3), pp.RandomChoice([pp.Rescaling1D(2.0)])]:
        with pytest.raises(ValueError, match="either signals or legacy data"):
            layer(tree, training=training)
