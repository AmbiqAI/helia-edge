"""State pair naming, records and manifest fields; runs without Keras or a training backend."""

import json

import pydantic
import pytest

from helia_edge.export import ExportManifest, ExportSpec, TensorRecord, TensorRole
from helia_edge.export.manifest import TensorEntry
from helia_edge.export.result import state_scales_tied
from helia_edge.export.spec import state_input_name, state_output_name, state_pair


@pytest.mark.parametrize(
    ("name", "pair"),
    [("state_in_0", ("in", 0)), ("state_out_12", ("out", 12)), ("state_in_01", None), ("state_in_", None)]
    + [("serving_default_state_in_0:0", None), ("state_out_0_1", None), ("cache_in_0", None), ("signal", None)],
)
def test_state_names_parse_strictly(name, pair):
    assert state_pair(name) == pair


def test_state_names_round_trip():
    assert state_pair(state_input_name(3)) == ("in", 3)
    assert state_pair(state_output_name(3)) == ("out", 3)


def record(name, role=TensorRole.SIGNAL, pair=None, scale=None, zero_point=None, dtype="float32"):
    return TensorRecord(name=name, role=role, shape=(1, 4), dtype=dtype, scale=scale, zero_point=zero_point, pair=pair)


def state(name, pair, scale=None, zero_point=None):
    dtype = "float32" if scale is None else "int16"
    return record(name, TensorRole.STATE, pair, scale, zero_point, dtype)


def test_tied_needs_every_integer_pair_to_share_scale_and_zero_point():
    signal = record("x", scale=0.5, zero_point=0, dtype="int16")
    tied = [state("a", 0, 0.1, 0), state("b", 1, 0.2, 3)]
    assert state_scales_tied([signal, *tied], [state("c", 1, 0.2, 3), state("d", 0, 0.1, 0)]) is True
    assert state_scales_tied(tied, [state("c", 1, 0.2, 3), state("d", 0, 0.1, 1)]) is False
    assert state_scales_tied(tied, [state("c", 1, 0.2, 3), state("d", 0, 0.11, 0)]) is False


def test_tied_is_none_without_integer_state():
    assert state_scales_tied([record("x", scale=0.5, zero_point=0, dtype="int8")], [record("y")]) is None
    assert state_scales_tied([state("a", 0)], [state("b", 0)]) is None


def test_unpaired_state_is_refused():
    with pytest.raises(ValueError, match="do not pair up"):
        state_scales_tied([state("a", 0, 0.1, 0)], [state("b", 1, 0.1, 0)])
    with pytest.raises(ValueError, match="'a' has no pair"):
        state_scales_tied([state("a", None, 0.1, 0)], [])


@pytest.mark.parametrize("tolerance", [-0.01, 1.0])
def test_tie_tolerance_is_a_fraction(tolerance):
    with pytest.raises(pydantic.ValidationError):
        ExportSpec(precision="a16w8", io_dtype="int16", mode="keras", state_tie_tolerance=tolerance)


def test_tensor_entries_carry_the_pair():
    entry = TensorEntry.from_record(state("state_in_0", 0, 0.1, 0))
    assert entry.pair == 0 and entry.role == TensorRole.STATE


def test_a_manifest_without_state_fields_still_reads(tmp_path):
    file = {"path": "m/model.tflite", "sha256": "0" * 64, "bytes": 1}
    tensor = {"name": "x", "role": "signal", "shape": [1, 4], "dtype": "float32", "scale": None, "zero_point": None}
    spec = {"format": "litert", "precision": "fp32", "io_dtype": "float32", "mode": "concrete", "strict": True}
    manifest = {
        "schema": "helia-edge/manifest@1",
        "recipe": {**file, "path": "recipe.yaml"},
        "environment": {"helia_edge": "0.1.0", "helia_edge_commit": None, "python": "3.12", "platform": "Linux"}
        | {"packages": {"numpy": "2.1.3"}},
        "entries": [{"name": "m", "spec": spec, "model": file, "inputs": [tensor], "outputs": [tensor]}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    entry = ExportManifest.read(path).entries[0]
    assert entry.state_scales_tied is None and entry.inputs[0].pair is None
    assert entry.spec is not None and entry.spec.state_tie_tolerance == 0.01
