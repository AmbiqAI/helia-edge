"""The ``helia-edge`` commands that run without Keras: schema, info, inspect and the refusals before a build."""

import dataclasses
import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from helia_edge.cli import app
from helia_edge.export import ExportRecord


def test_cli_prints_the_export_record_schema():
    result = CliRunner().invoke(app, ["export", "schema"])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.output)
    assert schema["title"] == "ExportRecord" and schema["properties"]["schema"]["const"] == "helia-edge/export-record@1"
    assert schema == ExportRecord.model_json_schema(by_alias=True)


@pytest.mark.parametrize("command", ["run", "verify"])
def test_the_recipe_commands_are_gone(command):
    result = CliRunner().invoke(app, ["export", command, "recipe.yaml"])
    assert result.exit_code == 2 and f"No such command '{command}'" in result.output


def test_cli_info_and_schema_do_not_import_training_frameworks():
    source = """
import json, sys
from typer.testing import CliRunner
from helia_edge.cli import app
for args in (["info"], ["export", "schema"], ["export", "reproduce", "missing.json", "--weights", "w.h5"]):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (3 if "reproduce" in args else 0), result.output
report = json.loads(CliRunner().invoke(app, ["info"]).output)
assert {"helia_edge", "helia_edge_source", "python", "packages", "keras_backend_env", "installed"} <= report.keys()
assert not {"keras", "tensorflow", "torch"} & sys.modules.keys(), sorted({"keras", "tensorflow", "torch"} & sys.modules.keys())
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_inspect_without_tensorflow_says_what_to_install(tmp_path):
    import importlib.util

    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("Checks the base environment")
    model = tmp_path / "m.tflite"
    model.write_bytes(b"TFL3")
    result = CliRunner().invoke(app, ["inspect", str(model)])
    assert result.exit_code != 0 and "helia-edge[litert]" in result.output


@pytest.mark.parametrize("content", [None, "{}", "not json"])
def test_reproduce_exits_3_for_a_missing_or_invalid_record(tmp_path, content):
    record = tmp_path / "record.json"
    if content is not None:
        record.write_text(content)
    result = CliRunner().invoke(app, ["export", "reproduce", str(record), "--weights", str(tmp_path / "w.h5")])
    assert result.exit_code == 3 and f"input: {record} is not a readable export record" in result.output


@pytest.mark.parametrize("source", ["local", "unknown"])
def test_create_with_require_provenance_refuses_code_a_record_cannot_identify(tmp_path, monkeypatch, source):
    from helia_edge.export import result

    record = dataclasses.replace(
        result.environment_record(), helia_edge="9.9.9", helia_edge_commit=None, helia_edge_source=source
    )
    monkeypatch.setattr(result, "environment_record", lambda: record)
    spec, weights = tmp_path / "spec.json", tmp_path / "w.weights.h5"
    spec.write_text("{}")
    weights.write_bytes(b"")
    args = ["export", "create", str(spec), "--weights", str(weights), "--precision", "fp32"]
    refused = CliRunner().invoke(app, [*args, "--out", str(tmp_path / "out"), "--require-provenance"])
    assert refused.exit_code == 1 and f"{source} install" in refused.stderr and not (tmp_path / "out").exists()


@pytest.mark.parametrize("batch_size", [0, 2**31])
def test_create_refuses_a_batch_size_outside_int32(tmp_path, batch_size):
    spec, weights = tmp_path / "spec.json", tmp_path / "w.weights.h5"
    spec.write_text("{}")
    weights.write_bytes(b"")
    args = ["export", "create", str(spec), "--weights", str(weights), "--precision", "fp32", "--out", str(tmp_path)]
    result = CliRunner().invoke(app, [*args, "--batch-size", str(batch_size)])
    assert result.exit_code == 2 and "--batch-size" in result.output
