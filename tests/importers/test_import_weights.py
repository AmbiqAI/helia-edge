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
        name="toy", source=SourcePin(uri="file://w", sha256=sha256, format="safetensors"), rows=rows, unused=unused
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
        ("unknown_weight", "weights matching 'gamma'"),
        ("mapped_and_unused", "both mapped and listed as unused"),
        ("unused_not_in_file", "not in the file"),
        ("overflow_on_cast", "not finite as float32"),
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
    elif change == "unused_not_in_file":
        unused = ("absent",)
    elif change == "overflow_on_cast":
        values["dense.bias_a"] = values["dense.bias_a"].astype(np.float64)
        values["dense.bias_b"] = values["dense.bias_b"].astype(np.float64)
        values["dense.bias_a"][0] = 1e39
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


def test_a_declared_unused_tensor_is_reported(source):
    values = tensors() | {"extra": np.zeros(2, np.float32)}
    path, sha256 = source(values)
    report = import_weights(model(), mapping(sha256, unused=("extra",)), path)
    assert report.unused == ("extra",)


@pytest.mark.parametrize(
    "parts",
    [
        [(2, 0), (2, 0)],  # one part twice, the other never
        [(2, 0), None],  # a part and the whole tensor
        [(2, 0), (4, 1)],  # different part counts
    ],
)
def test_inconsistent_split_layouts_are_refused(source, parts):
    values = tensors()
    values["packed"] = np.concatenate([values.pop("dense.bias_a"), values.pop("dense.bias_b")])
    x = keras.Input((3,))
    two = keras.Model(x, [keras.layers.Dense(4, name="dense")(x), keras.layers.Dense(4, name="other")(x)])
    values["head.weight"] = np.zeros((4, 3), np.float32)

    def bias_row(part, layer):
        transforms = () if part is None else (Split(axis=0, parts=part[0], index=part[1]),)
        return WeightRow(sources=("packed",), transforms=transforms, layer=layer, weight="bias")

    rows = (
        WeightRow(sources=("dense.weight",), transforms=(Transpose(perm=(1, 0)),), layer="dense", weight="kernel"),
        WeightRow(sources=("head.weight",), transforms=(Transpose(perm=(1, 0)),), layer="other", weight="kernel"),
        bias_row(parts[0], "dense"),
        bias_row(parts[1], "other"),
    )
    path, sha256 = source(values)
    with pytest.raises(ValueError, match="split parts are not each used once"):
        import_weights(two, mapping(sha256, rows), path)


class Counter(keras.layers.Layer):
    def __init__(self, count_dtype="int32", **kwargs):
        super().__init__(**kwargs)
        self.count_dtype = count_dtype

    def build(self, input_shape):
        self.count = self.add_weight(
            name="count", shape=(2,), dtype=self.count_dtype, initializer="zeros", trainable=False
        )

    def call(self, x):
        return x


def test_a_float_source_for_an_integer_weight_is_refused(source):
    x = keras.Input((3,))
    counted = keras.Model(x, Counter(name="counter")(x))
    path, sha256 = source({"count": np.array([1.5, 2.0], np.float32)})
    rows = (WeightRow(sources=("count",), layer="counter", weight="count"),)
    with pytest.raises(ValueError, match="source kinds"):
        import_weights(counted, mapping(sha256, rows), path)
    path, sha256 = source({"count": np.array([1, 2], np.int64)})
    import_weights(counted, mapping(sha256, rows), path)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(counted.get_layer("counter").count), [1, 2])


def test_a_file_changed_while_read_is_refused(source, monkeypatch):
    from helia_edge.importers import readers

    path, sha256 = source()
    original = readers.read_safetensors

    def read_then_change(p):
        result = original(p)
        p.write_bytes(p.read_bytes() + b" ")
        return result

    monkeypatch.setattr(readers, "read_safetensors", read_then_change)
    with pytest.raises(ValueError, match="changed while it was read"):
        import_weights(model(), mapping(sha256), path)


def test_torch_tensors_numpy_cannot_hold_are_refused(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("needs torch")
    import torch

    torch.save({"w": torch.zeros(2, dtype=torch.bfloat16)}, tmp_path / "bf16.pt")
    with pytest.raises(ValueError, match="'w' has dtype torch.bfloat16"):
        read_torch(tmp_path / "bf16.pt")


def test_onnx_weights_in_separate_files_are_refused(tmp_path):
    if importlib.util.find_spec("onnx") is None:
        pytest.skip("needs the onnx extra")
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        initializer=[numpy_helper.from_array(np.ones((2, 3), np.float32), "w")],
    )
    onnx.save(helper.make_model(graph), tmp_path / "m.onnx", save_as_external_data=True, size_threshold=0)
    assert any(p.name != "m.onnx" for p in tmp_path.iterdir())
    with pytest.raises(ValueError, match=r"stores initializers \['w'\] in separate data files"):
        read_onnx(tmp_path / "m.onnx")


def test_weights_inside_composite_layers_are_addressed_by_path(source):
    x = keras.Input((4, 8))
    mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, name="mha")
    attended = keras.Model(x, mha(x, x))
    rng = np.random.default_rng(5)
    values = {w.path: rng.normal(size=tuple(w.shape)).astype(np.float32) for w in attended.weights}
    rows = tuple(WeightRow(sources=(path,), layer="mha", weight=path.removeprefix("mha/")) for path in sorted(values))
    path, sha256 = source(values)
    import_weights(attended, mapping(sha256, rows), path)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(mha.query_dense.kernel), values["mha/query/kernel"])
    with pytest.raises(ValueError, match="has no sublayers"):
        import_weights(
            attended,
            mapping(sha256, (WeightRow(sources=("mha/query/kernel",), layer="mha/query", weight="kernel"),)),
            path,
        )


def test_bfloat16_weights_take_float_sources(source):
    x = keras.Input((3,))
    low = keras.Model(x, keras.layers.Dense(4, dtype="bfloat16", name="dense")(x))
    values = {k: v for k, v in tensors().items() if k != "head.weight"}
    path, sha256 = source(values)
    import_weights(low, mapping(sha256, ROWS[:2]), path)
    kernel = keras.ops.convert_to_numpy(keras.ops.cast(low.get_layer("dense").kernel, "float32"))
    np.testing.assert_allclose(kernel, values["dense.weight"].T, rtol=1e-2)


def test_values_are_checked_as_the_weight_stores_them(source):
    x = keras.Input((3,))
    low = keras.Model(x, keras.layers.Dense(4, dtype="bfloat16", name="dense")(x))
    values = {k: v for k, v in tensors().items() if k != "head.weight"}
    values["dense.weight"][0, 0] = 3.4e38  # finite in float32, inf in bfloat16
    path, sha256 = source(values)
    with pytest.raises(ValueError, match="not finite as bfloat16"):
        import_weights(low, mapping(sha256, ROWS[:2]), path)

    counted = keras.Model(x, Counter(count_dtype="int8", name="counter")(x))
    path, sha256 = source({"count": np.array([300, -200], np.int64)})
    with pytest.raises(ValueError, match="out of range for int8"):
        import_weights(
            counted, mapping(sha256, (WeightRow(sources=("count",), layer="counter", weight="count"),)), path
        )


def test_a_transform_with_a_bad_axis_is_listed_with_the_other_problems(source):
    rows = list(ROWS)
    rows[0] = WeightRow(
        sources=("dense.weight",), transforms=(Split(axis=5, parts=2, index=0),), layer="dense", weight="kernel"
    )
    path, sha256 = source()
    with pytest.raises(ValueError, match="- dense/kernel: tuple index out of range"):
        import_weights(model(), mapping(sha256, tuple(rows)), path)


def test_a_row_with_a_problem_is_not_also_listed_as_unmapped(source):
    rows = list(ROWS)
    rows[0] = WeightRow(sources=("dense.weight",), layer="dense", weight="kernel")  # shape mismatch
    path, sha256 = source()
    with pytest.raises(ValueError, match="mapped shape") as error:
        import_weights(model(), mapping(sha256, tuple(rows)), path)
    assert "without a mapping" not in str(error.value)
