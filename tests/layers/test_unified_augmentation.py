"""Joint geometry is checked against coordinate tensors and explicit parameters."""

import keras
import numpy as np
import pytest
from helia_edge.layers.preprocessing import (
    RandomCrop1D,
    RandomCrop2D,
    RandomFlip2D,
    BaseAugmentation1D,
    Normalization1D,
    FirFilter,
    RandomGaussianNoise1D,
    Sample,
)


def array(x):
    return keras.ops.convert_to_numpy(x)


def test_single_hierarchy():
    assert all(issubclass(c, BaseAugmentation1D) for c in (Normalization1D, FirFilter, RandomGaussianNoise1D))


@pytest.mark.parametrize("kind", ["crop1", "crop2", "flip"])
@pytest.mark.parametrize("channels_first", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_joint_coordinate_oracle(kind, channels_first, batched):
    shape = (2, 8, 2) if kind == "crop1" else (2, 6, 8, 2)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    fmt = "channels_first" if channels_first else "channels_last"
    if kind == "crop1":
        layer = RandomCrop1D(3, unique_batch=True, data_format=fmt)
        params = {"start": keras.ops.convert_to_tensor([1, 4], dtype="int32")}
        expected = np.stack([x[0, 1:4], x[1, 4:7]])
    elif kind == "crop2":
        layer = RandomCrop2D(2, 3, unique_batch=True, data_format=fmt)
        params = {
            "start_h": keras.ops.convert_to_tensor([1, 3], dtype="int32"),
            "start_w": keras.ops.convert_to_tensor([2, 4], dtype="int32"),
        }
        expected = np.stack([x[0, 1:3, 2:5], x[1, 3:5, 4:7]])
    else:
        layer = RandomFlip2D(horizontal=True, vertical=False, data_format=fmt)
        params = {"horizontal": keras.ops.convert_to_tensor([True, False])[:, None, None, None]}
        expected = np.stack([x[0, :, ::-1], x[1]])
    if channels_first:
        x = np.moveaxis(x, -1, 1)
        expected = np.moveaxis(expected, -1, 1)
    if not batched:
        x = x[0]
        expected = expected[0]
        params = {k: v[:1] for k, v in params.items()}
    sample = Sample(
        {"x": keras.ops.convert_to_tensor(x), "paired": keras.ops.convert_to_tensor(x + 1000)},
        {"seg": keras.ops.convert_to_tensor(x, dtype="int32")},
        {"valid": keras.ops.convert_to_tensor(x % 2 == 0)},
    )
    tree = sample.tensor_tree()
    out = layer(tree, training=True, transformations=params)
    np.testing.assert_array_equal(array(out["signals"]["x"]), expected)
    np.testing.assert_array_equal(array(out["signals"]["paired"]), expected + 1000)
    np.testing.assert_array_equal(array(out["targets"]["seg"]), expected.astype("int32"))
    np.testing.assert_array_equal(array(out["masks"]["valid"]), expected % 2 == 0)
    assert keras.backend.standardize_dtype(out["targets"]["seg"].dtype) == "int32"
    assert keras.backend.standardize_dtype(out["masks"]["valid"].dtype) == "bool"
    np.testing.assert_array_equal(array(tree["signals"]["x"]), x)
    np.testing.assert_array_equal(array(layer(tree, training=False)["signals"]["x"]), x)


def test_random_crop_repeatability_and_selection():
    a = RandomCrop1D(3, unique_batch=True, seed=42, aligned_targets=())
    b = RandomCrop1D(3, unique_batch=True, seed=42, aligned_targets=())
    x = keras.ops.reshape(keras.ops.arange(64, dtype="float32"), (8, 8, 1))
    inp = {"signals": {"x": x}, "targets": {"class": keras.ops.ones((8,), dtype="int32")}}
    a(inp, training=False)
    aa = a(inp, training=True)
    bb = b(inp, training=True)
    np.testing.assert_array_equal(array(aa["signals"]["x"]), array(bb["signals"]["x"]))
    np.testing.assert_array_equal(array(aa["targets"]["class"]), np.ones(8, dtype="int32"))
    assert len(set(array(aa["signals"]["x"])[:, 0, 0] % 8)) > 1


def test_alignment_rejects_mismatched_extent():
    with pytest.raises(ValueError, match="Aligned leaves"):
        RandomCrop1D(3)(
            {"signals": {"x": keras.ops.ones((2, 8, 1))}, "targets": {"y": keras.ops.ones((2, 7, 1))}}, training=True
        )


@pytest.mark.parametrize("kind", ["crop1", "flip"])
def test_joint_model_serialization_and_tf_data(tmp_path, kind):
    shape = (2, 8, 1) if kind == "crop1" else (2, 6, 8, 1)
    x = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    tree = Sample(
        {"x": keras.ops.convert_to_tensor(x)},
        {"seg": keras.ops.convert_to_tensor(x, dtype="int32")},
        {"valid": keras.ops.convert_to_tensor(x % 2 == 0)},
    ).tensor_tree()
    inputs = keras.tree.map_structure(
        lambda v: keras.Input(shape=v.shape[1:], dtype=keras.backend.standardize_dtype(v.dtype)), tree
    )
    layer = RandomCrop1D(3, unique_batch=True, seed=42) if kind == "crop1" else RandomFlip2D(seed=42)
    model = keras.Model(inputs, layer(inputs, training=True))
    model.save(tmp_path / "joint.keras")
    loaded = keras.models.load_model(tmp_path / "joint.keras")

    def aligned(out):
        np.testing.assert_array_equal(array(out["signals"]["x"]), array(out["targets"]["seg"]))
        np.testing.assert_array_equal(array(out["targets"]["seg"]) % 2 == 0, array(out["masks"]["valid"]))
        assert keras.backend.standardize_dtype(out["targets"]["seg"].dtype) == "int32"
        assert keras.backend.standardize_dtype(out["masks"]["valid"].dtype) == "bool"

    aligned(loaded(tree))
    if keras.backend.backend() == "tensorflow":
        import tensorflow as tf

        aligned(tf.function(lambda t: layer(t, training=True))(tree))
        aligned(next(iter(tf.data.Dataset.from_tensors(tree).map(lambda t: layer(t, training=True)))))
