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
    result = subprocess.run(  # outside the source tree, so the child imports the same helia_edge
        [sys.executable, "-c", source, *map(str, args)], cwd=tmp, text=True, capture_output=True, timeout=300
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
    no_model = edit_record(record, lambda data: data.update(model=None))
    changed = invoke("export", "reproduce", no_model, "--weights", weights, "--calibration", calibration)
    assert changed.exit_code == 3 and "has no model spec" in changed.output
    empty, huge = tmp / "empty.npy", tmp / "huge.npy"
    empty.write_bytes(b"")
    with open(huge, "wb") as file:  # a header that claims about 4 TB, and no data
        np.lib.format.write_array_header_1_0(file, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 1)})
    negative = tmp / "negative.npy"
    with open(negative, "wb") as file:
        np.lib.format.write_array_header_1_0(file, {"descr": "<f4", "fortran_order": False, "shape": (-1, 32, 4)})
    for samples in (empty, huge, negative):
        changed = invoke("export", "reproduce", record, "--weights", weights, "--calibration", samples)
        assert changed.exit_code == 3 and changed.output.strip().endswith("input"), changed.output
    garbage = tmp / "garbage.keras"
    garbage.write_bytes(b"not a zip")
    changed = invoke("export", "reproduce", record, "--weights", garbage, "--calibration", calibration)
    assert changed.exit_code == 3 and changed.output.strip().endswith("input"), changed.output

    binary = tmp / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00garbage")
    changed = invoke("export", "reproduce", binary, "--weights", weights)
    assert changed.exit_code == 3 and "is not a readable export record" in changed.output
    changed = invoke(
        "export",
        "reproduce",
        record,
        "--weights",
        weights,
        "--calibration",
        calibration,
        "--golden-inputs",
        calibration,
    )
    assert changed.exit_code == 3 and "--golden-inputs was given, but the record has no golden" in changed.output


def scale_input(data):
    data["io"]["inputs"][0]["scale"] *= 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["export"].update(io_dtype="float32"), "different: artifact: {"),
        (scale_input, "different: io"),
    ],
)
def test_a_record_whose_artifact_or_io_differs_exits_1(created, change, message):
    tmp, _, weights, calibration = created
    edited = edit_record(tmp / "out" / "record.json", change)
    result = invoke("export", "reproduce", edited, "--weights", weights, "--calibration", calibration)
    assert result.exit_code == 1 and message in result.output


def test_the_calibration_uri_is_carried_and_a_changed_artifact_beside_the_record_differs(created, tmp_path):
    import shutil

    tmp, _, weights, calibration = created
    out = shutil.copytree(tmp / "out", tmp_path / "out")
    uri = edit_record(
        out / "record.json", lambda data: data["export"]["calibration"].update(uri="https://example.com/c.npy")
    )
    same = invoke("export", "reproduce", uri, "--weights", weights, "--calibration", calibration)
    assert same.exit_code == 0, same.output
    (out / "model.tflite").write_bytes(b"TFL3 garbage")
    changed = invoke("export", "reproduce", out / "record.json", "--weights", weights, "--calibration", calibration)
    assert changed.exit_code == 1 and "different: model.tflite beside the record has 12 bytes" in changed.output
    (out / "model.tflite").chmod(0)
    try:
        unreadable = invoke(
            "export", "reproduce", out / "record.json", "--weights", weights, "--calibration", calibration
        )
    finally:
        (out / "model.tflite").chmod(0o644)
    assert unreadable.exit_code == 3 and "Permission denied" in unreadable.output


def test_a_failed_export_exits_4(created, monkeypatch):
    from helia_edge.export import api

    tmp, _, weights, calibration = created

    def broken(*args, **kwargs):
        raise RuntimeError("converter crashed")

    monkeypatch.setattr(api, "export", broken)
    result = invoke(
        "export", "reproduce", tmp / "out" / "record.json", "--weights", weights, "--calibration", calibration
    )
    assert result.exit_code == 4 and "converter crashed" in result.stderr and "nothing was compared" in result.stderr


def test_another_backend_cannot_export(created, tmp_path, monkeypatch):
    tmp, spec, weights, calibration = created
    monkeypatch.setattr(keras.backend, "backend", lambda: "torch")
    message = "LiteRT export needs KERAS_BACKEND=tensorflow"
    result = invoke(
        "export", "reproduce", tmp / "out" / "record.json", "--weights", weights, "--calibration", calibration
    )
    assert result.exit_code == 2 and f"environment: the Keras backend is 'torch'; {message}" in result.output
    result = invoke(
        "export", "create", tmp / "spec.yaml", "--weights", weights, "--precision", "fp32", "--out", tmp_path
    )
    assert result.exit_code == 1 and message in result.stderr and not (tmp_path / "record.json").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda environment: environment.update(python="0.0.0"), "python: 0.0.0 ->"),
        (lambda environment: environment.update(platform="Plan9-mips"), "platform: Plan9-mips ->"),
        (lambda environment: environment["helia_edge"].update(version="0.0.1"), "helia_edge.version: 0.0.1 ->"),
        (lambda environment: environment["helia_edge"].update(source="vcs"), "helia_edge.source: vcs ->"),
        (lambda environment: environment["helia_edge"].update(commit="abc123"), "helia_edge.commit: abc123 ->"),
        (lambda environment: environment["packages"].update(keras="0.0.0"), "keras: 0.0.0 ->"),
    ],
)
def test_another_environment_exits_2_unless_allowed(created, change, message):
    tmp, _, weights, calibration = created
    edited = edit_record(tmp / "out" / "record.json", lambda data: change(data["environment"]))
    args = ["export", "reproduce", edited, "--weights", weights, "--calibration", calibration]
    refused = invoke(*args)
    assert refused.exit_code == 2 and f"environment: {message}" in refused.output
    allowed = invoke(*args, "--allow-env-mismatch")
    assert allowed.exit_code == 0 and f"environment: {message}" in allowed.output


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
    reproduce = ["export", "reproduce", out / "record.json", "--weights", source]
    same = invoke(*reproduce, "--golden-inputs", calls)
    assert same.exit_code == 0 and same.output.strip().endswith("same"), same.output
    skipped = invoke(*reproduce)
    assert skipped.exit_code == 0 and "not compared: golden: pass --golden-inputs" in skipped.output
    uri = edit_record(
        out / "record.json", lambda data: data["golden"]["inputs"].update(uri="https://example.com/g.npy")
    )
    same = invoke("export", "reproduce", uri, "--weights", source, "--golden-inputs", calls)
    assert same.exit_code == 0, same.output
    no_model = edit_record(out / "record.json", lambda data: data.update(model=None))
    changed = invoke("export", "reproduce", no_model, "--weights", source)
    assert changed.exit_code == 3 and "has no model spec" in changed.output
    golden = (out / "golden.npz").read_bytes()
    (out / "golden.npz").write_bytes(golden[:-1] + bytes([golden[-1] ^ 1]))
    changed = invoke(*reproduce, "--golden-inputs", calls)
    assert changed.exit_code == 1 and "different: golden.npz beside the record" in changed.output
    (out / "golden.npz").write_bytes(golden)
    digest = edit_record(out / "record.json", lambda data: data["weights"].update(digest="sha256:" + "0" * 64))
    changed = invoke("export", "reproduce", digest, "--weights", source)
    assert changed.exit_code == 1 and "different: weights" in changed.output, changed.output
    resets = edit_record(out / "record.json", lambda data: data["golden"].update(resets=[4]))
    changed = invoke("export", "reproduce", resets, "--weights", source, "--golden-inputs", calls)
    assert changed.exit_code == 1 and "different: golden" in changed.output
    np.save(tmp_path / "other.npy", np.load(calls)[:6])
    changed = invoke(*reproduce, "--golden-inputs", tmp_path / "other.npy")
    assert changed.exit_code == 3 and "the record names" in changed.output
    write_safetensors(source, {name: value + 1 for name, value in tensors.items()})
    changed = invoke(*reproduce)
    assert changed.exit_code == 3 and f"the record names {sha256}" in changed.output
    # the family's mapping now pins the changed file; the record still names the file it was exported from
    other = hashlib.sha256(source.read_bytes()).hexdigest()
    pin = SourcePin(uri="file://silero.safetensors", sha256=other, format="safetensors")
    monkeypatch.setitem(silero_vad_params.MAPPINGS, mapping.name, mapping.model_copy(update={"source": pin}))
    changed = invoke(*reproduce)
    assert changed.exit_code == 3 and f"the record names {sha256}" in changed.output


def test_a_streaming_calibrated_export_reproduces_with_its_resets(tmp_path):
    spec = ModelSpec(params=SileroVadParams(stft="conv_blocks", magnitude="max_projection", encoder_tail="live_taps"))
    keras.utils.set_random_seed(0)
    model = build(spec, batch_size=1)
    rng = np.random.default_rng(3)
    for weight in model.weights:  # scaled so every layer stays in a useful range
        shape = tuple(weight.shape)
        scale = 4.0 if "basis" in weight.path else (0.1 if len(shape) == 1 else 1 / np.sqrt(np.prod(shape[:-1])))
        weight.assign((scale * rng.standard_normal(shape)).astype(np.float32))
    model.save_weights(tmp_path / "w.weights.h5")
    t = np.arange(512 * 48 + 64) / 16000
    signal = 0.05 * rng.standard_normal(t.size) + 0.4 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    np.save(tmp_path / "calls.npy", np.stack([signal[i * 512 : i * 512 + 576] for i in range(48)]))
    spec_file, out = tmp_path / "spec.json", tmp_path / "out"
    spec_file.write_text(spec.model_dump_json())
    args = ["--weights", tmp_path / "w.weights.h5", "--calibration", tmp_path / "calls.npy"]
    result = invoke("export", "create", spec_file, *args, "--precision", "a16w8", "--resets", 24, "--out", out)
    assert result.exit_code == 0, result.output + str(result.exception)
    record = ExportRecord.read(out / "record.json")
    assert record.export.io_dtype == "int16" and record.export.calibration.resets == (24,)
    same = invoke("export", "reproduce", out / "record.json", *args)
    assert same.exit_code == 0 and same.output.strip().endswith("same"), same.output
    moved = edit_record(out / "record.json", lambda data: data["export"]["calibration"].update(resets=[8]))
    changed = invoke("export", "reproduce", moved, *args)
    assert changed.exit_code == 1 and "different: artifact" in changed.output


@pytest.mark.parametrize(
    ("flags", "options", "batch", "io"),
    [
        (["--mode", "concrete"], {"mode": "concrete"}, 1, "float32"),
        (["--batch-size", 2], {"mode": "keras"}, 2, "float32"),
        (["--no-strict", "--state-tie-tolerance", 0.2], {"strict": False, "state_tie_tolerance": 0.2}, 1, "float32"),
        (["--precision", "fp16"], {}, 1, "float16"),
    ],
)
def test_create_passes_its_options(tmp_path, flags, options, batch, io):
    from helia_edge.models import compact_tcn_params

    spec = ModelSpec(params=compact_tcn_params(num_classes=2), input_shape=(32, 4))
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(spec.model_dump_json())
    weights = weights_file(spec, tmp_path / "w.weights.h5", seed=0)
    precision = [] if "--precision" in flags else ["--precision", "fp32"]
    result = invoke("export", "create", spec_file, "--weights", weights, *precision, "--out", tmp_path, *flags)
    assert result.exit_code == 0, result.output + str(result.exception)
    record = ExportRecord.read(tmp_path / "record.json")
    assert record.export.options.model_dump(include=set(options)) == options and record.export.io_dtype == io
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
        ("io", "is not a valid IODType"),
        ("golden resets", "--golden-resets needs --golden-inputs"),
        ("yaml", "is not valid YAML"),
        ("npy", "invalid .npy header"),
        ("npz", "is not a .npy array"),
        ("keras", "error: "),
    ],
)
def test_create_reports_a_refused_export(tmp_path, change, message):
    spec_file = tmp_path / ("spec.YML" if change == "yaml" else "spec.json")
    spec = ModelSpec(params=SileroVadParams())
    spec_file.write_text({"spec": "{}", "yaml": "params: [unclosed"}.get(change, spec.model_dump_json()))
    weights = weights_file(spec, tmp_path / "w.weights.h5", seed=0)
    with open(tmp_path / "negative.npy", "wb") as file:
        np.lib.format.write_array_header_1_0(file, {"descr": "<f4", "fortran_order": False, "shape": (-1, 576)})
    np.savez(tmp_path / "arrays.npz", x=np.zeros(3))
    if change == "keras":
        weights = tmp_path / "garbage.keras"
        weights.write_bytes(b"not a zip")
    flags = {
        "mapping": ["--mapping", "nope"],
        "precision": ["--precision", "a16w8"],
        "io": ["--io-dtype", ""],
        "golden resets": ["--golden-resets", 2],
        "npy": ["--golden-inputs", tmp_path / "negative.npy"],
        "npz": ["--golden-inputs", tmp_path / "arrays.npz"],
    }.get(change, [])
    precision = [] if change == "precision" else ["--precision", "fp32"]
    result = invoke("export", "create", spec_file, "--weights", weights, *precision, *flags, "--out", tmp_path / "out")
    assert result.exit_code == 1 and "error: " in result.stderr
    assert re.search(message, result.stderr), result.stderr
    assert not (tmp_path / "out").exists()


def test_a_record_batch_beyond_int32_is_invalid(tmp_path):
    from helia_edge.models import compact_tcn_params

    spec = ModelSpec(params=compact_tcn_params(num_classes=2), input_shape=(32, 4))
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(spec.model_dump_json())
    weights = weights_file(spec, tmp_path / "w.weights.h5", seed=0)
    result = invoke("export", "create", spec_file, "--weights", weights, "--precision", "fp32", "--out", tmp_path)
    assert result.exit_code == 0, result.output + str(result.exception)

    def batch_2_31(data):  # consistent with the I/O, so only the batch bound refuses it
        data["export"]["batch_size"] = 2**31
        for entry in (*data["io"]["inputs"], *data["io"]["outputs"]):
            entry["shape"][0] = 2**31

    edited = edit_record(tmp_path / "record.json", batch_2_31)
    changed = invoke("export", "reproduce", edited, "--weights", weights)
    assert changed.exit_code == 3 and "is not a readable export record" in changed.output
