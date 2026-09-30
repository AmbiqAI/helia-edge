"""Export recipe and manifest contracts and the Keras-free CLI commands; runs without Keras."""

import json
import subprocess
import sys

import pydantic
import pytest
from typer.testing import CliRunner

from helia_edge.cli import app
from helia_edge.export import BUILTIN_ARCHITECTURES, ExportManifest, ExportRecipe, load_recipe

SHA = "0" * 64


def recipe(**changes):
    data = {
        "schema": "helia-edge/export@1",
        "model": {"kind": "params_seed", "architecture": "mlperf_tiny", "params": {"architecture": "ad"}, "seed": 1},
        "calibration": {"source": {"kind": "array", "file": {"kind": "path", "path": "x.npy", "sha256": SHA}}},
        "exports": [
            {"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"},
            {"name": "a8w8", "precision": "a8w8", "io_dtype": "int8", "mode": "concrete"},
        ],
    }
    data.update(changes)
    return data


def test_a_valid_recipe_round_trips():
    parsed = ExportRecipe.model_validate(recipe())
    assert [e.name for e in parsed.exports] == ["fp32", "a8w8"]
    again = ExportRecipe.model_validate(json.loads(parsed.model_dump_json(by_alias=True)))
    assert again == parsed


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema": "helia-edge/export@2"}, "schema"),
        ({"exports": []}, "at least one export"),
        ({"calibration": None}, "need calibration"),
        (
            {"exports": [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"}]},
            "no export is calibrated",
        ),
        (
            {
                "exports": [{"name": "a", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"}] * 2,
                "calibration": None,
            },
            "unique",
        ),
        ({"exports": [{"name": "a", "precision": "fp32", "io_dtype": "int8", "mode": "concrete"}]}, "not valid for"),
        ({"exports": [{"name": "a", "precision": "fp32", "io_dtype": "float32"}], "calibration": None}, "mode"),
        ({"extra": 1}, "extra"),
        ({"model": {"kind": "params_seed", "architecture": "tcn", "params": {}}}, "seed"),
        ({"model": {"kind": "keras_file", "file": {"kind": "path", "path": "m.keras", "sha256": "abc"}}}, "sha256"),
        ({"model": {"kind": "keras_file", "file": {"kind": "url", "url": "ftp://x/m.keras", "sha256": SHA}}}, "url"),
        (
            {"model": {"kind": "tflite_import", "file": {"kind": "path", "path": "m.tflite", "sha256": SHA}}},
            "tflite_import",
        ),
        (
            {
                "exports": [{"name": "../x", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"}],
                "calibration": None,
            },
            "name",
        ),
    ],
)
def test_invalid_recipes_are_rejected(changes, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        ExportRecipe.model_validate(recipe(**changes))


def test_tflite_import_takes_no_exports():
    model = {"kind": "tflite_import", "file": {"kind": "path", "path": "m.tflite", "sha256": SHA}}
    parsed = ExportRecipe.model_validate(recipe(model=model, exports=[], calibration=None))
    assert parsed.exports == ()


def test_load_recipe_reads_yaml_and_json(tmp_path):
    import yaml

    (tmp_path / "r.yaml").write_text(yaml.safe_dump(recipe()))
    (tmp_path / "r.json").write_text(json.dumps(recipe()))
    assert load_recipe(tmp_path / "r.yaml") == load_recipe(tmp_path / "r.json")


def test_builtin_architectures_are_named_strings():
    assert set(BUILTIN_ARCHITECTURES) == {"tcn", "mlperf_tiny", "miniresnet_v1", "timeppg", "cornet"}
    assert all(":" in target for target in BUILTIN_ARCHITECTURES.values())


@pytest.mark.parametrize(("kind", "title"), [("recipe", "ExportRecipe"), ("manifest", "ExportManifest")])
def test_cli_prints_json_schemas(kind, title):
    result = CliRunner().invoke(app, ["export", "schema", "--kind", kind])
    assert result.exit_code == 0, result.output
    schema = json.loads(result.output)
    assert schema["title"] == title and "schema" in schema["properties"]
    assert schema == (ExportRecipe if kind == "recipe" else ExportManifest).model_json_schema(by_alias=True)


def test_cli_info_and_schema_do_not_import_training_frameworks():
    source = """
import json, sys
from typer.testing import CliRunner
from helia_edge.cli import app
for args in (["info"], ["export", "schema"], ["export", "schema", "--kind", "manifest"]):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
report = json.loads(CliRunner().invoke(app, ["info"]).output)
assert {"helia_edge", "python", "packages", "keras_backend", "installed"} <= report.keys()
assert not {"keras", "tensorflow", "torch"} & sys.modules.keys(), sorted({"keras", "tensorflow", "torch"} & sys.modules.keys())
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
