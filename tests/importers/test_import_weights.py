"""import_weights: every check refuses before anything is assigned; readers for ONNX and PyTorch files."""

import hashlib
import importlib.util

import keras
import numpy as np
import pytest

from helia_edge.importers import ImportReport, SourcePin, Split, Transpose, WeightMapping, WeightRow, import_weights
from helia_edge.importers.readers import read_onnx, read_torch


def model():
    keras.utils.set_random_seed(1)
    x = keras.Input((3,))
    y = keras.layers.Dense(4, name="dense")(x)
    return keras.Model(x, keras.layers.Dense(2, use_bias=False, name="head")(y))


def tensors():
    rng = np.random.default_rng(0)
    return {
        "dense.weight": rng.normal(size=(4, 3)).astype(np.float32),
        "dense.bias_a": rng.normal(size=(4,)).astype(np.float32),
        "dense.bias_b": rng.normal(size=(4,)).astype(np.float32),
        "head.weight": rng.normal(size=(2, 4)).astype(np.float32),
    }


ROWS = (
    WeightRow(sources=("dense.weight",), transforms=(Transpose(perm=(1, 0)),), layer="dense", weight="kernel"),
    WeightRow(sources=("dense.bias_a", "dense.bias_b"), combine="sum", layer="dense", weight="bias"),
    WeightRow(sources=("head.weight",), transforms=(Transpose(perm=(1, 0)),), layer="head", weight="kernel"),
)


@pytest.fixture
def source(tmp_path, write_safetensors):
    def write(values=None):
        path = tmp_path / "w.safetensors"
        write_safetensors(path, tensors() if values is None else values)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    return write


def mapping(sha256, rows=ROWS, unused=()):
    return WeightMapping(
        name="toy", format="safetensors", source=SourcePin(uri="file://w", sha256=sha256), rows=rows, unused=unused
    )


def weights(m):
    return [keras.ops.convert_to_numpy(w) for w in m.weights]


def test_every_weight_is_imported_through_its_transforms(source):
    path, sha256 = source()
    m = model()
    report = import_weights(m, mapping(sha256), path)
    t = tensors()
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(m.get_layer("dense").kernel), t["dense.weight"].T)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(m.get_layer("dense").bias), t["dense.bias_a"] + t["dense.bias_b"]
    )
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(m.get_layer("head").kernel), t["head.weight"].T)
    assert isinstance(report, ImportReport) and report.source_sha256 == sha256
    assert dict(report.assignments)["dense/bias"] == ("dense.bias_a", "dense.bias_b")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("sha256", "has sha256"),
        ("unused_source", "neither mapped nor listed as unused"),
        ("missing_source", "not in the file"),
        ("duplicate_source", "used more than once"),
        ("unassigned_weight", "model weights without a mapping"),
        ("shape", "mapped shape"),
        ("non_finite", "not finite"),
        ("kind", "source kinds"),
        ("unknown_layer", "No such layer|head2"),
        ("unknown_weight", "weights named 'gamma'"),
        ("mapped_and_unused", "both mapped and listed as unused"),
    ],
)
def test_a_failed_check_assigns_nothing(source, change, message):
    values, rows, unused = tensors(), list(ROWS), ()
    if change == "unused_source":
        values["extra"] = np.zeros(2, np.float32)
    elif change == "missing_source":
        rows[2] = WeightRow(sources=("head.w",), transforms=(Transpose(perm=(1, 0)),), layer="head", weight="kernel")
    elif change == "duplicate_source":
        rows[2] = WeightRow(
            sources=("dense.weight",), transforms=(Transpose(perm=(1, 0)),), layer="head", weight="kernel"
        )
        values.pop("head.weight")
    elif change == "unassigned_weight":
        rows = rows[:2]
        unused = ("head.weight",)
    elif change == "shape":
        rows[0] = WeightRow(sources=("dense.weight",), layer="dense", weight="kernel")
    elif change == "non_finite":
        values["dense.bias_a"][0] = np.nan
    elif change == "kind":
        values["head.weight"] = values["head.weight"].astype(np.int64)
    elif change == "unknown_layer":
        rows[2] = WeightRow(sources=("head.weight",), layer="head2", weight="kernel")
    elif change == "unknown_weight":
        rows[2] = WeightRow(sources=("head.weight",), layer="head", weight="gamma")
    elif change == "mapped_and_unused":
        unused = ("head.weight",)
    path, sha256 = source(values)
    m = model()
    before = weights(m)
    with pytest.raises(ValueError, match=message):
        import_weights(m, mapping("f" * 64 if change == "sha256" else sha256, tuple(rows), unused), path)
    for got, want in zip(weights(m), before, strict=True):
        np.testing.assert_array_equal(got, want)


def test_split_parts_must_each_be_used_once(source):
    values = tensors()
    values["packed"] = np.concatenate([values.pop("dense.bias_a"), values.pop("dense.bias_b")])
    rows = list(ROWS)
    rows[1] = WeightRow(
        sources=("packed",), transforms=(Split(axis=0, parts=2, index=0),), layer="dense", weight="bias"
    )
    path, sha256 = source(values)
    with pytest.raises(ValueError, match="split parts are not each used once"):
        import_weights(model(), mapping(sha256, tuple(rows)), path)

    x = keras.Input((3,))
    two = keras.Model(x, [keras.layers.Dense(4, name="dense")(x), keras.layers.Dense(4, name="other")(x)])
    rows = (
        WeightRow(sources=("dense.weight",), transforms=(Transpose(perm=(1, 0)),), layer="dense", weight="kernel"),
        WeightRow(sources=("packed",), transforms=(Split(axis=0, parts=2, index=0),), layer="dense", weight="bias"),
        WeightRow(sources=("head.weight",), transforms=(Transpose(perm=(1, 0)),), layer="other", weight="kernel"),
        WeightRow(sources=("packed",), transforms=(Split(axis=0, parts=2, index=1),), layer="other", weight="bias"),
    )
    values["head.weight"] = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
    path, sha256 = source(values)
    import_weights(two, mapping(sha256, rows), path)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(two.get_layer("other").bias), values["packed"][4:])


def test_onnx_initializers_are_read(tmp_path):
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("needs the onnx extra")
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[numpy_helper.from_array(weight, "w")],
    )
    onnx.save(helper.make_model(graph), tmp_path / "m.onnx")
    read = read_onnx(tmp_path / "m.onnx")
    assert list(read) == ["w"]
    np.testing.assert_array_equal(read["w"], weight)


def test_torch_state_dicts_are_read(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("needs torch")
    import torch

    state = {"layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "layer.bias": torch.zeros(2)}
    torch.save(state, tmp_path / "m.pt")
    read = read_torch(tmp_path / "m.pt")
    np.testing.assert_array_equal(read["layer.weight"], np.arange(6, dtype=np.float32).reshape(2, 3))
    torch.save({"nested": {"a": torch.zeros(1)}}, tmp_path / "nested.pt")
    with pytest.raises(ValueError, match="not a flat state dict"):
        read_torch(tmp_path / "nested.pt")
