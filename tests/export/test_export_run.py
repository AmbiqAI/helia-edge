"""Run recipes into manifests and verify them on the TensorFlow backend."""

import hashlib
import json

import keras
import numpy as np
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)
pytest.importorskip("ai_edge_litert.interpreter")

import yaml  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from helia_edge.cli import app  # noqa: E402
from helia_edge.export import ExportManifest, run_recipe, verify_manifest  # noqa: E402
from helia_edge.export.run import SourceError  # noqa: E402
from helia_edge.models import compact_tcn_params  # noqa: E402

TCN = compact_tcn_params(filters=8).model_dump(mode="json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_source(path):
    return {"kind": "path", "path": path.name, "sha256": sha(path)}


def array_source(path, key=None):
    source = {"kind": "array", "file": path_source(path)}
    return source if key is None else {**source, "key": key}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIA_EDGE_CACHE", str(tmp_path / "cache"))
    rng = np.random.default_rng(0)
    np.save(tmp_path / "ad.npy", rng.standard_normal((6, 640)).astype(np.float32))
    np.savez(tmp_path / "tcn.npz", x=rng.standard_normal((6, 32, 4)).astype(np.float32))
    return tmp_path


def write(path, data):
    path.write_text(yaml.safe_dump(data))
    return path


def ad_recipe(workdir, **changes):
    data = {
        "schema": "helia-edge/export@1",
        "model": {"kind": "params_seed", "architecture": "mlperf_tiny", "params": {"architecture": "ad"}, "seed": 3},
        "calibration": {"source": array_source(workdir / "ad.npy"), "samples": 4},
        "reference": {"source": array_source(workdir / "ad.npy"), "samples": 2},
        "exports": [
            {"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"},
            {"name": "a8w8", "precision": "a8w8", "io_dtype": "int8", "mode": "concrete"},
        ],
    }
    data.update(changes)
    return write(workdir / "ad.yaml", data)


def test_run_writes_models_references_and_manifest(workdir):
    manifest = run_recipe(ad_recipe(workdir), workdir / "out")
    assert ExportManifest.read(workdir / "out" / "manifest.json") == manifest
    assert manifest.recipe.path == "../ad.yaml" and manifest.recipe.sha256 == sha(workdir / "ad.yaml")
    for entry in manifest.entries:
        for record in (entry.model, entry.reference.inputs, entry.reference.outputs):
            assert sha(workdir / "out" / record.path) == record.sha256
    fp32, a8w8 = manifest.entries
    assert fp32.inputs[0].dtype == "float32" and a8w8.inputs[0].dtype == "int8"
    assert a8w8.inputs[0].scale is not None and fp32.inputs[0].role == "signal"
    inputs = np.load(workdir / "out" / a8w8.reference.inputs.path)
    assert inputs.dtype == np.int8 and inputs.shape == (2, 640)
    assert manifest.environment.packages["tensorflow"]


def test_run_is_reproducible_and_verifies(workdir):
    recipe = ad_recipe(workdir)
    first = run_recipe(recipe, workdir / "a")
    second = run_recipe(recipe, workdir / "b")
    assert [e.model.sha256 for e in first.entries] == [e.model.sha256 for e in second.entries]
    report = verify_manifest(workdir / "a" / "manifest.json")
    assert (report.status, report.differences) == ("ok", [])


def test_verify_detects_a_changed_export_file(workdir):
    manifest = run_recipe(ad_recipe(workdir), workdir / "out")
    model = workdir / "out" / manifest.entries[0].model.path
    model.write_bytes(model.read_bytes() + b"\0")
    report = verify_manifest(workdir / "out" / "manifest.json")
    assert report.status == "drift" and "fp32: model file" in report.differences[0]


def test_verify_detects_a_changed_recipe(workdir):
    run_recipe(ad_recipe(workdir), workdir / "out")
    ad_recipe(
        workdir,
        model={"kind": "params_seed", "architecture": "mlperf_tiny", "params": {"architecture": "ad"}, "seed": 4},
    )
    report = verify_manifest(workdir / "out" / "manifest.json")
    assert report.status == "drift" and "recipe ../ad.yaml changed" in report.differences


def test_verify_detects_regenerated_bytes_that_differ(workdir, monkeypatch):
    run_recipe(ad_recipe(workdir), workdir / "out")
    from helia_edge.export import run

    build = run.build_model

    def perturbed(source, base_dir):
        model = build(source, base_dir)
        model.weights[0].assign(model.weights[0] + 1e-2)
        return model

    monkeypatch.setattr(run, "build_model", perturbed)
    report = verify_manifest(workdir / "out" / "manifest.json")
    assert report.status == "drift"
    assert {d.split(" sha256")[0] for d in report.differences} >= {"fp32: model", "a8w8: model"}


def test_verify_refuses_a_different_environment(workdir, monkeypatch):
    run_recipe(ad_recipe(workdir), workdir / "out")
    manifest_path = workdir / "out" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["environment"]["packages"]["tensorflow"] = "0.0.1"
    manifest_path.write_text(json.dumps(data))
    report = verify_manifest(manifest_path)
    assert report.status == "env_mismatch" and report.environment_differences[0].startswith("tensorflow: 0.0.1 ->")
    labelled = verify_manifest(manifest_path, allow_env_mismatch=True)
    assert labelled.status == "ok" and labelled.environment_differences


def test_a_source_with_the_wrong_sha256_is_refused(workdir):
    recipe = ad_recipe(workdir)
    (workdir / "ad.npy").write_bytes((workdir / "ad.npy").read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(SourceError, match="expected"):
        run_recipe(recipe, workdir / "out")


def test_npz_sources_need_a_key(workdir):
    base = {
        "schema": "helia-edge/export@1",
        "model": {
            "kind": "params_seed",
            "architecture": "tcn",
            "params": TCN,
            "input_shape": [32, 4],
            "num_classes": 2,
            "seed": 1,
        },
        "exports": [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "keras"}],
    }
    write(workdir / "tcn.yaml", {**base, "reference": {"source": array_source(workdir / "tcn.npz")}})
    with pytest.raises(SourceError, match="set key"):
        run_recipe(workdir / "tcn.yaml", workdir / "out")
    write(workdir / "tcn.yaml", {**base, "reference": {"source": array_source(workdir / "tcn.npz", "x")}})
    manifest = run_recipe(workdir / "tcn.yaml", workdir / "out")
    assert manifest.entries[0].inputs[0].shape == (1, 32, 4)


def test_params_weights_keras_file_and_tflite_import(workdir):
    from helia_edge.export.architectures import resolve_architecture

    # Build in a fresh session, as run_recipe does, so layer names (and tensor names) match.
    keras.backend.clear_session()
    keras.utils.set_random_seed(9)
    trained = resolve_architecture("tcn")(TCN, (32, 4), 2)
    trained.save_weights(workdir / "tcn.weights.h5")
    trained.save(workdir / "tcn.keras")
    exports = [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"}]
    params = {"kind": "params_weights", "architecture": "tcn", "params": TCN, "input_shape": [32, 4], "num_classes": 2}
    write(
        workdir / "w.yaml",
        {
            "schema": "helia-edge/export@1",
            "exports": exports,
            "model": {**params, "weights": path_source(workdir / "tcn.weights.h5")},
        },
    )
    write(
        workdir / "k.yaml",
        {
            "schema": "helia-edge/export@1",
            "exports": exports,
            "model": {"kind": "keras_file", "file": path_source(workdir / "tcn.keras")},
        },
    )
    from_weights = run_recipe(workdir / "w.yaml", workdir / "w")
    from_keras = run_recipe(workdir / "k.yaml", workdir / "k")
    assert from_weights.entries[0].model.sha256 == from_keras.entries[0].model.sha256
    tflite = workdir / "w" / from_weights.entries[0].model.path
    (workdir / "m.tflite").write_bytes(tflite.read_bytes())
    write(
        workdir / "i.yaml",
        {
            "schema": "helia-edge/export@1",
            "model": {"kind": "tflite_import", "file": path_source(workdir / "m.tflite")},
        },
    )
    imported = run_recipe(workdir / "i.yaml", workdir / "i")
    assert imported.entries[0].name == "import" and imported.entries[0].spec is None
    assert imported.entries[0].model.sha256 == from_weights.entries[0].model.sha256
    assert verify_manifest(workdir / "i" / "manifest.json").status == "ok"


@pytest.mark.parametrize(
    ("architecture", "params", "shape", "classes", "message"),
    [
        ("mlperf_tiny", {"architecture": "kws"}, [49, 10, 1], None, "fixed input shape"),
        ("tcn", TCN, None, None, "needs input_shape"),
        ("miniresnet_v1", {}, [32, 20, 1], None, "needs num_classes"),
        ("timeppg", {}, [256, 4], 3, "has no num_classes"),
        ("unknown", {}, None, None, "Unknown architecture"),
    ],
)
def test_architecture_arguments_are_checked(workdir, architecture, params, shape, classes, message):
    model = {"kind": "params_seed", "architecture": architecture, "params": params, "seed": 0}
    if shape is not None:
        model["input_shape"] = shape
    if classes is not None:
        model["num_classes"] = classes
    write(
        workdir / "r.yaml",
        {
            "schema": "helia-edge/export@1",
            "model": model,
            "exports": [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"}],
        },
    )
    with pytest.raises(ValueError, match=message):
        run_recipe(workdir / "r.yaml", workdir / "out")


def test_only_selects_exports(workdir):
    manifest = run_recipe(ad_recipe(workdir), workdir / "out", only=["fp32"])
    assert [e.name for e in manifest.entries] == ["fp32"]
    assert verify_manifest(workdir / "out" / "manifest.json").status == "ok"
    with pytest.raises(ValueError, match="Unknown export names"):
        run_recipe(ad_recipe(workdir), workdir / "out", only=["nope"])


def test_cli_run_verify_and_inspect(workdir):
    runner = CliRunner()
    recipe = ad_recipe(workdir)
    result = runner.invoke(app, ["export", "run", str(recipe), "--out", str(workdir / "out")])
    assert result.exit_code == 0, result.output
    manifest = workdir / "out" / "manifest.json"
    assert runner.invoke(app, ["export", "verify", str(manifest)]).exit_code == 0
    model = workdir / "out" / "a8w8" / "model.tflite"
    report = json.loads(runner.invoke(app, ["inspect", str(model)]).output)
    assert report["inputs"][0]["dtype"] == "int8" and "FULLY_CONNECTED" in report["operators"]
    model.write_bytes(model.read_bytes() + b"\0")
    drift = runner.invoke(app, ["export", "verify", str(manifest)])
    assert drift.exit_code == 1 and "drift: a8w8: model file" in drift.output
    model.write_bytes(model.read_bytes()[:-1])
    data = json.loads(manifest.read_text())
    data["environment"]["python"] = "0.0.0"
    manifest.write_text(json.dumps(data))
    mismatch = runner.invoke(app, ["export", "verify", str(manifest)])
    assert mismatch.exit_code == 2 and "environment: python: 0.0.0" in mismatch.output


def test_bytes_do_not_depend_on_models_built_earlier_in_the_process(workdir):
    recipe = {
        "schema": "helia-edge/export@1",
        "model": {
            "kind": "params_seed",
            "architecture": "tcn",
            "params": TCN,
            "input_shape": [32, 4],
            "num_classes": 2,
            "seed": 1,
        },
        "exports": [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "keras"}],
    }
    write(workdir / "tcn.yaml", recipe)
    first = run_recipe(workdir / "tcn.yaml", workdir / "a").entries[0].model.sha256
    inputs = keras.Input((4,))
    keras.Model(inputs, keras.layers.Dense(2)(keras.layers.Dense(3)(inputs)))  # advances Keras name counters
    assert run_recipe(workdir / "tcn.yaml", workdir / "b").entries[0].model.sha256 == first


def test_verify_reports_a_changed_source_as_drift(workdir):
    run_recipe(ad_recipe(workdir), workdir / "out")
    data = np.load(workdir / "ad.npy")
    np.save(workdir / "ad.npy", data + 1)
    report = verify_manifest(workdir / "out" / "manifest.json")
    assert report.status == "drift" and report.differences[0].startswith("source:")


def test_verify_detects_edited_manifest_records(workdir):
    run_recipe(ad_recipe(workdir), workdir / "out")
    path = workdir / "out" / "manifest.json"
    data = json.loads(path.read_text())
    data["entries"][1]["inputs"][0]["scale"] = 123.0
    path.write_text(json.dumps(data))
    report = verify_manifest(path)
    assert report.status == "drift" and "a8w8: recorded inputs differs from the regenerated one" in report.differences


def test_npz_sources_are_recognised_by_content(workdir):
    (workdir / "tcn.bin").write_bytes((workdir / "tcn.npz").read_bytes())
    recipe = {
        "schema": "helia-edge/export@1",
        "model": {
            "kind": "params_seed",
            "architecture": "tcn",
            "params": TCN,
            "input_shape": [32, 4],
            "num_classes": 2,
            "seed": 1,
        },
        "reference": {"source": array_source(workdir / "tcn.bin", "x")},
        "exports": [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "keras"}],
    }
    write(workdir / "tcn.yaml", recipe)
    assert run_recipe(workdir / "tcn.yaml", workdir / "out").entries[0].reference is not None


def test_multi_input_keras_files_are_refused_clearly(workdir):
    a, b = keras.Input((4,), batch_size=1), keras.Input((4,), batch_size=1)
    keras.Model([a, b], keras.layers.Add()([a, b])).save(workdir / "two.keras")
    write(
        workdir / "two.yaml",
        {
            "schema": "helia-edge/export@1",
            "model": {"kind": "keras_file", "file": path_source(workdir / "two.keras")},
            "exports": [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "concrete"}],
        },
    )
    with pytest.raises(ValueError, match="single-input"):
        run_recipe(workdir / "two.yaml", workdir / "out")


def test_only_import_selects_the_imported_model(workdir):
    manifest = run_recipe(ad_recipe(workdir), workdir / "src", only=["fp32"])
    (workdir / "m.tflite").write_bytes((workdir / "src" / manifest.entries[0].model.path).read_bytes())
    write(
        workdir / "i.yaml",
        {
            "schema": "helia-edge/export@1",
            "model": {"kind": "tflite_import", "file": path_source(workdir / "m.tflite")},
        },
    )
    assert [e.name for e in run_recipe(workdir / "i.yaml", workdir / "i", only=["import"]).entries] == ["import"]


def test_calibration_uses_the_first_rows_in_stored_order(workdir):
    rng = np.random.default_rng(5)
    first = rng.standard_normal((4, 640)).astype(np.float32)
    np.save(workdir / "first.npy", first)
    np.save(workdir / "mixed.npy", np.concatenate([first, 50 * rng.standard_normal((4, 640)).astype(np.float32)]))
    exports = [{"name": "a8w8", "precision": "a8w8", "io_dtype": "int8", "mode": "concrete"}]
    model = {"kind": "params_seed", "architecture": "mlperf_tiny", "params": {"architecture": "ad"}, "seed": 3}

    def run(name, source, samples):
        calibration = {"source": array_source(workdir / source)} | ({"samples": samples} if samples else {})
        recipe = {"schema": "helia-edge/export@1", "model": model, "calibration": calibration, "exports": exports}
        return run_recipe(write(workdir / f"{name}.yaml", recipe), workdir / name).entries[0].model.sha256

    assert run("head", "mixed.npy", 4) == run("only", "first.npy", None) != run("all", "mixed.npy", None)
    with pytest.raises(SourceError, match="has 4 rows"):
        run("over", "first.npy", 5)


def test_params_weights_and_keras_file_verify(workdir):
    keras.backend.clear_session()
    keras.utils.set_random_seed(9)
    inputs = keras.Input((32, 4))  # batch None: the recipe rebuilds it with batch size 1
    from helia_edge.models import TcnModel, TcnParams

    model = TcnModel.model_from_params(inputs, TcnParams.from_config(TCN), num_classes=2)
    model.save(workdir / "free.keras")
    model.save_weights(workdir / "free.weights.h5")
    exports = [{"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "keras"}]
    params = {"kind": "params_weights", "architecture": "tcn", "params": TCN, "input_shape": [32, 4], "num_classes": 2}
    recipes = {
        "k": {"model": {"kind": "keras_file", "file": path_source(workdir / "free.keras")}},
        "w": {"model": {**params, "weights": path_source(workdir / "free.weights.h5")}},
    }
    for name, recipe in recipes.items():
        write(workdir / f"{name}.yaml", {"schema": "helia-edge/export@1", "exports": exports, **recipe})
        manifest = run_recipe(workdir / f"{name}.yaml", workdir / name)
        assert manifest.entries[0].inputs[0].shape == (1, 32, 4)
        assert verify_manifest(workdir / name / "manifest.json").status == "ok"


def test_verify_compares_regenerated_reference_outputs(workdir, monkeypatch):
    run_recipe(ad_recipe(workdir), workdir / "out")
    from helia_edge.export.runner import LiteRTRunner

    run = LiteRTRunner.run
    monkeypatch.setattr(LiteRTRunner, "run", lambda self, x: run(self, x) + 1)
    report = verify_manifest(workdir / "out" / "manifest.json")
    assert report.status == "drift" and any("reference outputs sha256" in d for d in report.differences)


def test_url_sources_are_downloaded_once_and_cached(workdir):
    import functools
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    (workdir / "served").mkdir()
    (workdir / "served" / "ad.npy").write_bytes((workdir / "ad.npy").read_bytes())
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(workdir / "served"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = {"kind": "url", "url": f"http://127.0.0.1:{server.server_port}/ad.npy", "sha256": sha(workdir / "ad.npy")}
    recipe = ad_recipe(workdir, reference={"source": {"kind": "array", "file": url}, "samples": 2})
    try:
        first = run_recipe(recipe, workdir / "a")
    finally:
        server.shutdown()
    assert (workdir / "cache" / "sources" / url["sha256"]).is_file()
    second = run_recipe(recipe, workdir / "b")  # server is down: the cached file is used
    assert [e.model.sha256 for e in first.entries] == [e.model.sha256 for e in second.entries]


def test_manifest_records_the_environment(workdir):
    import platform

    import tensorflow as tf

    environment = run_recipe(ad_recipe(workdir), workdir / "out").environment
    assert environment.python == platform.python_version()
    assert environment.platform == f"{platform.system()}-{platform.machine()}"
    assert environment.packages["tensorflow"] == tf.__version__ and environment.packages["keras"] == keras.__version__
    assert set(environment.packages) == {"numpy", "keras", "tensorflow", "ai-edge-litert"}


def test_cli_verify_exit_codes_for_invalid_manifests_and_combined_drift(workdir):
    runner = CliRunner()
    missing = runner.invoke(app, ["export", "verify", str(workdir / "missing.json")])
    assert missing.exit_code == 3 and "invalid manifest" in missing.output
    (workdir / "bad.json").write_text("{}")
    assert runner.invoke(app, ["export", "verify", str(workdir / "bad.json")]).exit_code == 3
    run_recipe(ad_recipe(workdir), workdir / "out")
    manifest = workdir / "out" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["environment"]["python"] = "0.0.0"
    manifest.write_text(json.dumps(data))
    model = workdir / "out" / "fp32" / "model.tflite"
    model.write_bytes(model.read_bytes() + b"\0")
    both = runner.invoke(app, ["export", "verify", str(manifest)])
    assert (
        both.exit_code == 1 and "environment: python: 0.0.0" in both.output and "drift: fp32: model file" in both.output
    )
