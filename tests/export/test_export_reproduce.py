"""``helia-edge export create`` and ``export reproduce``: a record re-exports the same bytes, and says why not."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)

from helia_edge.cli import app
from helia_edge.export import ExportRecord
from helia_edge.importers import SourcePin
from helia_edge.models import SileroVadParams, TcnParams
from helia_edge.models.spec import ModelSpec, build

KIT_SPECS = json.loads((Path(__file__).parents[1] / "fixtures" / "kit-tcn-specs.json").read_text())


def invoke(*args):
    return CliRunner().invoke(app, [str(arg) for arg in args])


def operator_names(content):
    from ai_edge_litert.interpreter import Interpreter

    return {op["op_name"] for op in Interpreter(model_content=content)._get_ops_details()}


def weights_file(spec, path, seed):
    keras.utils.set_random_seed(seed)
    build(spec).save_weights(path)
    return path


def edit_record(path, change):
    data = json.loads(path.read_text())
    change(data)
    edited = path.with_name("edited.json")
    edited.write_text(json.dumps(data))
    return edited


@pytest.fixture(scope="module")
def created(tmp_path_factory):
    """A kit TCN (heartKIT PPG denoise, dynamic batch as trained) exported a8w8 by ``export create``."""
    tmp = tmp_path_factory.mktemp("create")
    entry = KIT_SPECS["heartkit_ppg_denoise"]
    spec = ModelSpec(params=TcnParams.model_validate(entry["params"]), input_shape=entry["input_shape"])
    spec_file = tmp / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec.model_dump(mode="json")))
    weights = weights_file(spec, tmp / "trained.weights.h5", seed=0)
    calibration = tmp / "calibration.npy"
    np.save(calibration, np.random.default_rng(1).normal(size=(8, *spec.input_shape)))  # float64, cast to float32
    args = ["--weights", weights, "--precision", "a8w8", "--calibration", calibration, "--out", tmp / "out"]
    result = invoke("export", "create", spec_file, *args)
    assert result.exit_code == 0, result.output + str(result.exception)
    return tmp, spec, weights, calibration


def test_create_exports_a_static_batch_without_space_to_batch(created):
    tmp, spec, weights, calibration = created
    record = ExportRecord.read(tmp / "out" / "record.json")
    content = (tmp / "out" / "model.tflite").read_bytes()
    assert record.artifact.sha256 == hashlib.sha256(content).hexdigest() and record.model == spec
    assert not {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"} & operator_names(content)
    assert record.export.batch_size == 1 and all(entry.shape[0] == 1 for entry in record.io.inputs)
    assert record.export.io_dtype == "int8" and record.io.inputs[0].dtype == "int8"


def test_a_created_record_reproduces_in_a_new_process(created):
    tmp, _, weights, calibration = created
    args = ["export", "reproduce", tmp / "out" / "record.json", "--weights", weights, "--calibration", calibration]
    source = "import sys; from helia_edge.cli import app; sys.argv[0] = 'helia-edge'; app()"
    result = subprocess.run(
        [sys.executable, "-c", source, *map(str, args)], text=True, capture_output=True, timeout=300
    )
    assert result.returncode == 0 and result.stdout.strip().endswith("same"), result.stdout + result.stderr


def test_other_weights_or_calibration_exit_3(created):
    tmp, spec, weights, calibration = created
    record = tmp / "out" / "record.json"
    other = weights_file(spec, tmp / "other.weights.h5", seed=5)
    changed = invoke("export", "reproduce", record, "--weights", other, "--calibration", calibration)
    assert changed.exit_code == 3 and "digest" in changed.output
    samples = tmp / "other.npy"
    np.save(samples, np.load(calibration)[:4])
    changed = invoke("export", "reproduce", record, "--weights", weights, "--calibration", samples)
    assert changed.exit_code == 3 and "the record names" in changed.output
    missing = invoke("export", "reproduce", record, "--weights", weights)
    assert missing.exit_code == 3 and "pass its .npy file" in missing.output


def test_a_record_whose_artifact_differs_exits_1(created):
    tmp, _, weights, calibration = created
    edited = edit_record(tmp / "out" / "record.json", lambda data: data["export"].update(io_dtype="float32"))
    result = invoke("export", "reproduce", edited, "--weights", weights, "--calibration", calibration)
    assert result.exit_code == 1 and "different: artifact: sha256" in result.output


def test_another_environment_exits_2_unless_allowed(created):
    tmp, _, weights, calibration = created
    edited = edit_record(tmp / "out" / "record.json", lambda data: data["environment"].update(python="0.0.0"))
    args = ["export", "reproduce", edited, "--weights", weights, "--calibration", calibration]
    refused = invoke(*args)
    assert refused.exit_code == 2 and "environment: python: 0.0.0 ->" in refused.output
    allowed = invoke(*args, "--allow-env-mismatch")
    assert allowed.exit_code == 0 and "environment: python: 0.0.0 ->" in allowed.output


def test_imported_weights_and_a_golden_reproduce(tmp_path, write_safetensors, silero_tensors, monkeypatch):
    from helia_edge.models import silero_vad_params

    spec = ModelSpec(params=SileroVadParams(stft="conv_blocks", magnitude="max_projection", encoder_tail="live_taps"))
    tensors = silero_tensors()
    source = tmp_path / "silero.safetensors"
    write_safetensors(source, tensors)
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    mapping = silero_vad_params.SILERO_VAD_V6_ONNX.model_copy(
        update={"source": SourcePin(uri="file://silero.safetensors", sha256=sha256, format="safetensors")}
    )
    monkeypatch.setitem(silero_vad_params.MAPPINGS, mapping.name, mapping)
    spec_file, calls = tmp_path / "spec.json", tmp_path / "calls.npy"
    spec_file.write_text(spec.model_dump_json())
    np.save(calls, np.random.default_rng(2).normal(scale=0.1, size=(12, 576)))
    out = tmp_path / "out"
    args = ["--weights", source, "--mapping", mapping.name, "--precision", "fp32", "--out", out]
    result = invoke("export", "create", spec_file, *args, "--golden-inputs", calls, "--golden-resets", 6)
    assert result.exit_code == 0, result.output + str(result.exception)
    record = ExportRecord.read(out / "record.json")
    assert record.weights.import_.mapping == mapping.name and record.weights.import_.source.sha256 == sha256
    assert record.golden.steps == 12 and record.golden.resets == (6,) and (out / "golden.npz").exists()
    same = invoke("export", "reproduce", out / "record.json", "--weights", source, "--golden-inputs", calls)
    assert same.exit_code == 0 and same.output.strip().endswith("same"), same.output
    write_safetensors(source, {name: value + 1 for name, value in tensors.items()})
    changed = invoke("export", "reproduce", out / "record.json", "--weights", source)
    assert changed.exit_code == 3 and f"the record names {sha256}" in changed.output


@pytest.mark.parametrize(
    ("flags", "mode", "batch"), [(["--mode", "concrete"], "concrete", 1), (["--batch-size", 2], "keras", 2)]
)
def test_create_passes_the_mode_and_batch_size(tmp_path, flags, mode, batch):
    from helia_edge.models import compact_tcn_params

    spec = ModelSpec(params=compact_tcn_params(num_classes=2), input_shape=(32, 4))
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(spec.model_dump_json())
    weights = weights_file(spec, tmp_path / "w.weights.h5", seed=0)
    result = invoke(
        "export", "create", spec_file, "--weights", weights, "--precision", "fp32", "--out", tmp_path, *flags
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    record = ExportRecord.read(tmp_path / "record.json")
    assert record.export.options.mode == mode and record.export.io_dtype == "float32"
    assert record.export.batch_size == batch and record.io.inputs[0].shape[0] == batch
    same = invoke("export", "reproduce", tmp_path / "record.json", "--weights", weights)
    assert same.exit_code == 0, same.output


@pytest.mark.parametrize(("source", "commit"), [("vcs", "abc123"), ("release", None), ("local", None)])
def test_create_records_the_install_and_warns_unless_it_identifies_the_code(tmp_path, monkeypatch, source, commit):
    import dataclasses

    from helia_edge.export import litert, result
    from helia_edge.models import compact_tcn_params

    record = dataclasses.replace(
        result.environment_record(), helia_edge="9.9.9", helia_edge_commit=commit, helia_edge_source=source
    )
    monkeypatch.setattr(result, "environment_record", lambda: record)
    monkeypatch.setattr(litert, "environment_record", lambda: record)
    spec = ModelSpec(params=compact_tcn_params(num_classes=2), input_shape=(32, 4))
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(spec.model_dump_json())
    weights = weights_file(spec, tmp_path / "w.weights.h5", seed=0)
    args = ["export", "create", spec_file, "--weights", weights, "--precision", "fp32"]
    created = invoke(*args, "--out", tmp_path / "out")
    assert created.exit_code == 0, created.output + str(created.exception)
    assert ("does not identify its code" in created.stderr) == (source == "local")
    helia_edge = ExportRecord.read(tmp_path / "out" / "record.json").environment.helia_edge
    assert (helia_edge.version, helia_edge.source, helia_edge.commit) == ("9.9.9", source, commit)
    published = invoke(*args, "--out", tmp_path / "published", "--require-provenance")
    assert published.exit_code == (1 if source == "local" else 0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("mapping", "has no weight mapping 'nope'; it has \\['silero_vad_v6_onnx'\\]"),
        ("spec", "validation error"),
        ("precision", "calibration"),
    ],
)
def test_create_reports_a_refused_export(tmp_path, change, message):
    spec_file = tmp_path / "spec.json"
    spec = ModelSpec(params=SileroVadParams())
    spec_file.write_text("{}" if change == "spec" else spec.model_dump_json())
    weights = weights_file(spec, tmp_path / "w.weights.h5", seed=0)
    flags = {"mapping": ["--mapping", "nope"], "spec": [], "precision": ["--precision", "a16w8"]}[change]
    precision = [] if change == "precision" else ["--precision", "fp32"]
    result = invoke("export", "create", spec_file, "--weights", weights, *precision, *flags, "--out", tmp_path / "out")
    assert result.exit_code == 1 and "error: " in result.stderr
    assert re.search(message, result.stderr), result.stderr
    assert not (tmp_path / "out").exists()
