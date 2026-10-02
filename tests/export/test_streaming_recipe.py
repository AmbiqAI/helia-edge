"""Recipes for streaming models: stream calibration, golden@2 sequences, imported weights, Silero registration."""

import hashlib
import importlib.util
import json
import zipfile

import keras
import numpy as np
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)
pytest.importorskip("ai_edge_litert.interpreter")

from helia_edge import registry  # noqa: E402
from helia_edge.export import ExportManifest, run_recipe, verify_manifest  # noqa: E402
from helia_edge.export.architectures import resolve_architecture  # noqa: E402
from helia_edge.export.recipe import ParamsImport  # noqa: E402
from helia_edge.export.run import SourceError, build_model  # noqa: E402
from helia_edge.importers import SourcePin, Transpose, WeightMapping, WeightRow  # noqa: E402
from helia_edge.layers import StreamingLSTMCell, state_input, state_output  # noqa: E402

FEATURES, UNITS = 6, 8


def build_toy_stream(params, input_shape, num_classes):
    x = keras.Input((FEATURES,), batch_size=1, name="signal")
    h, c = state_input(0, (UNITS,), 1), state_input(1, (UNITS,), 1)
    h_next, c_next = StreamingLSTMCell(UNITS, name="lstm")([x, h, c])
    prob = keras.layers.Dense(1, activation="sigmoid", name="prob")(h_next)
    return keras.Model([x, h, c], [prob, state_output(0, h_next), state_output(1, c_next)])


@pytest.fixture(autouse=True)
def toy_stream_architecture(monkeypatch):
    """Register the toy architecture for each test only, so other tests see the built-in registry."""
    monkeypatch.setitem(registry.architectures._values, "toy_stream", build_toy_stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_source(path):
    return {"kind": "array", "file": {"kind": "path", "path": path.name, "sha256": sha(path)}}


EXPORTS = [
    {"name": "fp32", "precision": "fp32", "io_dtype": "float32", "mode": "keras"},
    {"name": "a16w8", "precision": "a16w8", "io_dtype": "int16", "mode": "keras"},
]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIA_EDGE_CACHE", str(tmp_path / "cache"))
    rng = np.random.default_rng(0)
    np.save(tmp_path / "cal.npy", rng.normal(size=(64, FEATURES)).astype(np.float32))
    np.save(tmp_path / "ref.npy", rng.normal(size=(20, FEATURES)).astype(np.float32))
    return tmp_path


def write(path, recipe):
    path.write_text(json.dumps(recipe))
    return path


def stream_recipe(workdir, model=None, calibration_resets=(32,), reference_resets=(10,)):
    return {
        "schema": "helia-edge/export@1",
        "model": model or {"kind": "params_seed", "architecture": "toy_stream", "params": {}, "seed": 3},
        "calibration": {"source": array_source(workdir / "cal.npy"), "resets": list(calibration_resets)},
        "reference": {"source": array_source(workdir / "ref.npy"), "resets": list(reference_resets)},
        "exports": EXPORTS,
    }


def golden_arrays(out, entry):
    with np.load(out / entry.reference.golden.file.path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def test_a_streaming_recipe_writes_sequence_goldens(workdir):
    manifest = run_recipe(write(workdir / "r.json", stream_recipe(workdir)), workdir / "out")
    for entry in manifest.entries:
        golden = entry.reference.golden
        assert entry.reference.inputs is None and entry.reference.outputs is None
        assert (golden.kind, golden.steps, golden.resets, golden.resolver) == ("sequence", 20, (10,), "builtin_ref")
        assert golden.source.uri == "path:ref.npy" and golden.source.sha256 == sha(workdir / "ref.npy")
        arrays = golden_arrays(workdir / "out", entry)
        assert sorted(arrays) == ["input_0", "input_1", "input_2", "output_0", "output_1", "output_2"]
        # Keys follow subgraph order, the order of the manifest's tensor records
        for i, record in enumerate(entry.inputs):
            assert arrays[f"input_{i}"].shape == (20, *record.shape)
            assert arrays[f"input_{i}"].dtype == np.dtype(record.dtype.value)
        for i, record in enumerate(entry.outputs):
            assert arrays[f"output_{i}"].shape == (20, *record.shape)
        pairs = {
            r.pair: (f"input_{i}", f"output_{j}")
            for i, r in enumerate(entry.inputs)
            for j, o in enumerate(entry.outputs)
            if r.pair is not None and r.pair == o.pair
        }
        assert sorted(pairs) == [0, 1]
        for state_in, state_out in pairs.values():
            carried, produced = arrays[state_in], arrays[state_out]
            np.testing.assert_array_equal(carried[1:10], produced[:9])
            np.testing.assert_array_equal(carried[11:], produced[10:-1])
            zero = [r.zero_point or 0 for r in entry.inputs if f"input_{entry.inputs.index(r)}" == state_in][0]
            assert (carried[[0, 10]] == zero).all()
    assert manifest.entries[1].state_scales_tied is True
    assert ExportManifest.read(workdir / "out" / "manifest.json") == manifest


def test_goldens_are_stored_uncompressed(workdir):
    manifest = run_recipe(write(workdir / "r.json", stream_recipe(workdir)), workdir / "out")
    for entry in manifest.entries:
        with zipfile.ZipFile(workdir / "out" / entry.reference.golden.file.path) as archive:
            assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}


def test_the_signal_rows_are_the_reference_as_fed(workdir):
    manifest = run_recipe(write(workdir / "r.json", stream_recipe(workdir)), workdir / "out")
    entry = manifest.entries[0]
    (signal,) = [i for i, r in enumerate(entry.inputs) if r.pair is None]
    np.testing.assert_array_equal(
        golden_arrays(workdir / "out", entry)[f"input_{signal}"][:, 0], np.load(workdir / "ref.npy")
    )


def test_verify_regenerates_goldens_and_detects_a_changed_one(workdir):
    write(workdir / "r.json", stream_recipe(workdir))
    manifest = run_recipe(workdir / "r.json", workdir / "out")
    assert verify_manifest(workdir / "out" / "manifest.json").status == "ok"
    path = workdir / "out" / manifest.entries[1].reference.golden.file.path
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    report = verify_manifest(workdir / "out" / "manifest.json")
    assert report.status == "drift" and any("golden" in d for d in report.differences)


@pytest.mark.skipif(importlib.util.find_spec("helia_model_zoo") is None, reason="needs helia-model-zoo")
def test_goldens_pass_the_model_zoo_check(workdir):
    from helia_model_zoo import golden

    manifest = run_recipe(write(workdir / "r.json", stream_recipe(workdir)), workdir / "out")
    for entry in manifest.entries:
        record = entry.reference.golden
        problems = golden.check(
            workdir / "out" / entry.model.path,
            workdir / "out" / record.file.path,
            kind=record.kind,
            steps=record.steps,
            resets=record.resets,
            resolver=record.resolver,
            replay=True,
        )
        assert problems == [], problems


@pytest.mark.parametrize(
    ("calibration", "reference", "message"),
    [((64,), (10,), "Calibration resets"), ((32,), (5, 3), "Reference resets"), ((32,), (20,), "Reference resets")],
)
def test_resets_must_be_increasing_steps_within_the_sequence(workdir, calibration, reference, message):
    write(workdir / "r.json", stream_recipe(workdir, calibration_resets=calibration, reference_resets=reference))
    with pytest.raises(ValueError, match=message):
        run_recipe(workdir / "r.json", workdir / "out")


def test_resets_apply_to_streaming_models_only(workdir):
    np.save(workdir / "x.npy", np.random.default_rng(1).normal(size=(4, 8, 8, 1)).astype(np.float32))
    recipe = {
        "schema": "helia-edge/export@1",
        "model": {"kind": "params_seed", "architecture": "mlperf_tiny", "params": {"architecture": "ad"}, "seed": 1},
        "reference": {"source": array_source(workdir / "x.npy"), "resets": [2]},
        "exports": [EXPORTS[0]],
    }
    with pytest.raises(ValueError, match="streaming models only"):
        run_recipe(write(workdir / "r.json", recipe), workdir / "out")


def toy_mapping(sha256):
    rows = (
        WeightRow(sources=("lstm.w",), transforms=(Transpose(perm=(1, 0)),), layer="lstm", weight="kernel"),
        WeightRow(sources=("lstm.r",), transforms=(Transpose(perm=(1, 0)),), layer="lstm", weight="recurrent_kernel"),
        WeightRow(sources=("lstm.b",), layer="lstm", weight="bias"),
        WeightRow(sources=("prob.w",), transforms=(Transpose(perm=(1, 0)),), layer="prob", weight="kernel"),
        WeightRow(sources=("prob.b",), layer="prob", weight="bias"),
    )
    return WeightMapping(name="toy", format="safetensors", source=SourcePin(uri="file://toy", sha256=sha256), rows=rows)


def test_params_import_builds_and_imports_the_pinned_weights(workdir, write_safetensors, monkeypatch):
    rng = np.random.default_rng(2)
    tensors = {
        "lstm.w": rng.normal(size=(4 * UNITS, FEATURES)).astype(np.float32),
        "lstm.r": rng.normal(size=(4 * UNITS, UNITS)).astype(np.float32),
        "lstm.b": rng.normal(size=(4 * UNITS,)).astype(np.float32),
        "prob.w": rng.normal(size=(1, UNITS)).astype(np.float32),
        "prob.b": np.array([0.5], np.float32),
    }
    write_safetensors(workdir / "toy.safetensors", tensors)
    pinned = sha(workdir / "toy.safetensors")
    monkeypatch.setitem(registry.weight_mappings._values, "toy_mapping", toy_mapping(pinned))
    model = {
        "kind": "params_import",
        "architecture": "toy_stream",
        "params": {},
        "mapping": "toy_mapping",
        "weights": {"kind": "path", "path": "toy.safetensors", "sha256": pinned},
    }
    manifest = run_recipe(write(workdir / "r.json", stream_recipe(workdir, model=model)), workdir / "out")
    built = build_model(ParamsImport.model_validate(model), workdir)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(built.get_layer("lstm").kernel), tensors["lstm.w"].T)
    assert verify_manifest(workdir / "out" / "manifest.json").status == "ok"
    assert len(manifest.entries) == 2

    other = {**model, "weights": {**model["weights"], "sha256": "f" * 64}}
    with pytest.raises(SourceError, match="is not the file mapping 'toy_mapping' is pinned to"):
        run_recipe(write(workdir / "r2.json", stream_recipe(workdir, model=other)), workdir / "out2")
    unknown = {**model, "mapping": "absent"}
    with pytest.raises(ValueError, match="absent"):
        run_recipe(write(workdir / "r3.json", stream_recipe(workdir, model=unknown)), workdir / "out3")


def test_multi_input_models_must_stream_with_batch_size_one(workdir):
    a, b = keras.Input((4,), batch_size=1, name="a"), keras.Input((4,), batch_size=1, name="b")
    keras.Model([a, b], keras.layers.Add()([a, b])).save(workdir / "two.keras")
    x, h = keras.Input((FEATURES,), name="signal"), keras.Input((UNITS,), name="state_in_0")
    stream = keras.Model([x, h], [keras.layers.Dense(1)(x), state_output(0, keras.layers.Dense(UNITS)(h))])
    stream.save(workdir / "stream.keras")
    for name, message in (
        ("two.keras", "single-input models and streaming models"),
        ("stream.keras", "batch size 1 on every input"),
    ):
        recipe = {
            "schema": "helia-edge/export@1",
            "model": {"kind": "keras_file", "file": {"kind": "path", "path": name, "sha256": sha(workdir / name)}},
            "exports": [EXPORTS[0]],
        }
        with pytest.raises(ValueError, match=message):
            run_recipe(write(workdir / "r.json", recipe), workdir / "out")


def test_a_reference_of_one_step_takes_no_resets(workdir):
    np.save(workdir / "one.npy", np.zeros((1, FEATURES), np.float32))
    recipe = stream_recipe(workdir) | {"reference": {"source": array_source(workdir / "one.npy"), "resets": [1]}}
    with pytest.raises(ValueError, match="absent for a single step"):
        run_recipe(write(workdir / "r.json", recipe), workdir / "out")


def test_silero_vad_is_a_registered_architecture():
    from helia_edge.models import SileroVadParams

    model = resolve_architecture("vad_silero_v6")({"name": "vad"}, None, None)
    assert model.name == "vad" and [t.name for t in model.inputs] == ["audio", "state_in_0", "state_in_1"]
    assert SileroVadParams().samples == 576
    from helia_edge.models import silero_vad_v6

    assert silero_vad_v6("named").name == "named"
    with pytest.raises(ValueError, match="context"):
        resolve_architecture("vad_silero_v6")({"context": 32}, None, None)
    with pytest.raises(ValueError, match="fixed input shape"):
        resolve_architecture("vad_silero_v6")({}, (576,), None)
    assert "silero_vad_v6_onnx" in registry.weight_mappings


def test_calibration_resets_reach_stream_calibration(workdir, monkeypatch):
    from helia_edge.export import api

    seen = []
    original = api.stream_calibration
    monkeypatch.setattr(
        api,
        "stream_calibration",
        lambda model, signals, resets=(): seen.append(tuple(resets)) or original(model, signals, resets),
    )
    run_recipe(write(workdir / "r.json", stream_recipe(workdir, calibration_resets=(16, 40))), workdir / "out")
    assert seen == [(16, 40)]


def test_verify_detects_changed_golden_records(workdir):
    write(workdir / "r.json", stream_recipe(workdir))
    run_recipe(workdir / "r.json", workdir / "out")
    path = workdir / "out" / "manifest.json"
    data = json.loads(path.read_text())
    data["entries"][0]["reference"]["golden"]["resets"] = [11]
    path.write_text(json.dumps(data))
    report = verify_manifest(path)
    assert report.status == "drift" and any("recorded golden differs" in d for d in report.differences)


def test_resets_of_an_imported_tflite_model_are_checked(workdir):
    write(workdir / "r.json", stream_recipe(workdir))
    manifest = run_recipe(workdir / "r.json", workdir / "out")
    (workdir / "stream.tflite").write_bytes((workdir / "out" / manifest.entries[0].model.path).read_bytes())
    recipe = {
        "schema": "helia-edge/export@1",
        "model": {
            "kind": "tflite_import",
            "file": {"kind": "path", "path": "stream.tflite", "sha256": sha(workdir / "stream.tflite")},
        },
        "reference": {"source": array_source(workdir / "ref.npy"), "resets": [25]},
    }
    with pytest.raises(ValueError, match="Reference resets"):
        run_recipe(write(workdir / "i.json", recipe), workdir / "imported")
    recipe["reference"]["resets"] = [5]
    imported = run_recipe(write(workdir / "i.json", recipe), workdir / "imported")
    assert imported.entries[0].reference.golden.resets == (5,)


@pytest.mark.skipif(importlib.util.find_spec("helia_model_zoo") is None, reason="needs helia-model-zoo")
def test_a_seeded_silero_model_exports_through_a_recipe(workdir):
    from helia_model_zoo import golden

    rng = np.random.default_rng(4)
    np.save(workdir / "audio.npy", (0.1 * rng.standard_normal((6, 576))).astype(np.float32))
    recipe = {
        "schema": "helia-edge/export@1",
        "model": {"kind": "params_seed", "architecture": "vad_silero_v6", "params": {}, "seed": 5},
        "reference": {"source": array_source(workdir / "audio.npy"), "resets": [3]},
        "exports": [EXPORTS[0]],
    }
    manifest = run_recipe(write(workdir / "s.json", recipe), workdir / "out")
    (entry,) = manifest.entries
    assert sorted(r.pair for r in entry.inputs if r.pair is not None) == [0, 1]
    assert entry.reference.golden.steps == 6
    record = entry.reference.golden
    problems = golden.check(
        workdir / "out" / entry.model.path,
        workdir / "out" / record.file.path,
        kind=record.kind,
        steps=record.steps,
        resets=record.resets,
        resolver=record.resolver,
    )
    assert problems == [], problems
