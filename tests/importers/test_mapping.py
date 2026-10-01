"""Weight mapping schema, transforms and the safetensors reader; runs without Keras or a training backend."""

import numpy as np
import pydantic
import pytest

from helia_edge.importers import GateReorder, Reshape, SourcePin, Split, Transpose, WeightMapping, WeightRow
from helia_edge.importers.readers import read_safetensors
from helia_edge.registry import importers

PIN = SourcePin(uri="file://model.bin", sha256="0" * 64)


def test_transforms_match_numpy():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.testing.assert_array_equal(Transpose(perm=(2, 0, 1)).apply(x), np.transpose(x, (2, 0, 1)))
    np.testing.assert_array_equal(Reshape(shape=(6, 4)).apply(x), x.reshape(6, 4))
    np.testing.assert_array_equal(Split(axis=2, parts=2, index=1).apply(x), x[:, :, 2:])


def test_gate_reorder_moves_whole_gate_blocks():
    gates = {g: np.full((2, 3), i, np.float32) for i, g in enumerate("iofc")}
    stacked = np.concatenate([gates[g] for g in "iofc"])
    reordered = GateReorder(source="iofc", target="ifco").apply(stacked)
    np.testing.assert_array_equal(reordered, np.concatenate([gates[g] for g in "ifco"]))
    with pytest.raises(ValueError, match="does not split into 4 gates"):
        GateReorder(source="iofc", target="ifco").apply(np.zeros((6, 3)))


@pytest.mark.parametrize(("source", "target"), [("iofc", "ifcx"), ("iofc", "ifc"), ("iioo", "iioo")])
def test_gate_orders_must_name_the_same_distinct_gates(source, target):
    with pytest.raises(pydantic.ValidationError, match="same distinct gates"):
        GateReorder(source=source, target=target)


def test_rows_combine_and_transform_in_order():
    row = WeightRow(sources=("a", "b"), combine="sum", transforms=(Transpose(perm=(1, 0)),), layer="l", weight="w")
    a, b = np.ones((2, 3), np.float32), np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(row.value({"a": a, "b": b}), (a + b).T)
    concat = WeightRow(sources=("a", "b"), combine="concat", concat_axis=1, layer="l", weight="w")
    np.testing.assert_array_equal(concat.value({"a": a, "b": b}), np.concatenate([a, b], axis=1))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"sources": ("a", "b")}, "need combine"),
        ({"sources": ("a",), "combine": "sum"}, "combine applies to several sources"),
        ({"sources": ("a", "a"), "combine": "sum"}, "repeat"),
        ({"sources": ("a",), "transforms": (Transpose(perm=(1, 0)), Split(axis=0, parts=2, index=0))}, "split must be"),
        (
            {"sources": ("a", "b"), "combine": "concat", "transforms": (Split(axis=0, parts=2, index=0),)},
            "split must be",
        ),
    ],
)
def test_inconsistent_rows_are_refused(row, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        WeightRow(layer="l", weight="w", **row)


def test_a_weight_mapped_twice_is_refused():
    rows = (WeightRow(sources=("a",), layer="l", weight="w"), WeightRow(sources=("b",), layer="l", weight="w"))
    with pytest.raises(pydantic.ValidationError, match="mapped more than once"):
        WeightMapping(name="m", format="safetensors", source=PIN, rows=rows)


def test_mappings_round_trip_through_json():
    rows = (
        WeightRow(
            sources=("w",),
            transforms=(GateReorder(source="iofc", target="ifco"), Transpose(perm=(1, 0))),
            layer="lstm",
            weight="kernel",
        ),
        WeightRow(sources=("p",), transforms=(Split(axis=0, parts=2, index=1),), layer="dense", weight="bias"),
    )
    mapping = WeightMapping(name="m", format="onnx", source=PIN, rows=rows, unused=("extra",))
    assert WeightMapping.model_validate_json(mapping.model_dump_json()) == mapping


def test_safetensors_files_read_back(tmp_path, write_safetensors):
    tensors = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.array([1.5, -2.0], np.float16),
        "c": np.array([[7]], np.int64),
    }
    write_safetensors(tmp_path / "w.safetensors", tensors)
    read = read_safetensors(tmp_path / "w.safetensors")
    assert sorted(read) == ["a", "b", "c"]
    for name, value in tensors.items():
        np.testing.assert_array_equal(read[name], value)
        assert read[name].dtype == value.dtype


def test_the_importer_registry_has_the_built_in_formats():
    assert {"onnx", "safetensors", "torch"} <= set(importers)
    assert importers.get("safetensors") is read_safetensors
