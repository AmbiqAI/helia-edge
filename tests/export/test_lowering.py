"""Export lowerings: a registered rebuild of an architecture's model before conversion (Silero VAD v6 for an NPU)."""

import hashlib
import importlib.util
import json

import keras
import numpy as np
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)
pytest.importorskip("ai_edge_litert.interpreter")

from helia_edge.export import ExportSpec, export_model, run_recipe, verify_manifest  # noqa: E402
from helia_edge.export.litert import operator_names  # noqa: E402
from helia_edge.models import silero_vad_v6  # noqa: E402
from helia_edge.registry import NotRegistered  # noqa: E402

FP32 = {"precision": "fp32", "io_dtype": "float32", "mode": "keras"}


def test_a_lowering_needs_the_architecture():
    with pytest.raises(ValueError, match="architecture"):
        export_model(silero_vad_v6(), ExportSpec(**FP32, lowering="npu"))


def test_an_unregistered_lowering_is_refused():
    with pytest.raises(NotRegistered, match="vad_silero_v6:gpu"):
        export_model(silero_vad_v6(), ExportSpec(**FP32, lowering="gpu"), architecture="vad_silero_v6")
    with pytest.raises(NotRegistered, match="tcn:npu"):
        export_model(silero_vad_v6(), ExportSpec(**FP32, lowering="npu"), architecture="tcn")


def array_source(path):
    return {
        "kind": "array",
        "file": {"kind": "path", "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


def random_silero_weights(path):
    """Random Silero weights scaled so every layer stays in a useful range; a seeded build's output is constant."""
    model = silero_vad_v6()
    rng = np.random.default_rng(7)
    for weight in model.weights:
        shape = tuple(weight.shape)
        scale = 0.1 if len(shape) == 1 else 1 / np.sqrt(np.prod(shape[:-1]))
        scale = 4.0 if "basis" in weight.path else scale
        weight.assign((scale * rng.standard_normal(shape)).astype(np.float32))
    model.save_weights(path)


def silero_npu_recipe(tmp_path):
    random_silero_weights(tmp_path / "silero.weights.h5")
    weights = {"kind": "path", "path": "silero.weights.h5"}
    weights["sha256"] = hashlib.sha256((tmp_path / "silero.weights.h5").read_bytes()).hexdigest()
    rng = np.random.default_rng(6)
    t = np.arange(512 * 136 + 64) / 16000
    speechy = 0.3 * np.sin(2 * np.pi * 400 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0) + 0.02 * rng.standard_normal(t.size)
    calls = np.stack([speechy[i * 512 : i * 512 + 576] for i in range(136)]).astype(np.float32)
    np.save(tmp_path / "cal.npy", calls[:128])
    np.save(tmp_path / "ref.npy", calls[128:])
    recipe = {
        "schema": "helia-edge/export@1",
        "model": {"kind": "params_weights", "architecture": "vad_silero_v6", "params": {}, "weights": weights},
        "calibration": {"source": array_source(tmp_path / "cal.npy"), "resets": [64]},
        "reference": {"source": array_source(tmp_path / "ref.npy"), "resets": [4]},
        "exports": [
            {"name": "fp32", **FP32},
            {"name": "fp32-npu", **FP32, "lowering": "npu"},
            {"name": "int16x8-npu", "precision": "a16w8", "io_dtype": "int16", "mode": "keras", "lowering": "npu"},
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(recipe))
    return tmp_path / "r.json"


@pytest.fixture
def silero_npu(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIA_EDGE_CACHE", str(tmp_path / "cache"))
    out = tmp_path / "out"
    return out, run_recipe(silero_npu_recipe(tmp_path), out)


def outputs(out, entry, key):
    with np.load(out / entry.reference.golden.file.path, allow_pickle=False) as arrays:
        return arrays[key]


def test_a_recipe_exports_silero_lowered_for_an_npu(silero_npu):
    out, manifest = silero_npu
    semantic, lowered, int16 = manifest.entries
    assert (semantic.spec.lowering, lowered.spec.lowering, int16.spec.lowering) == (None, "npu", "npu")
    assert int16.spec.strict is True and int16.state_scales_tied is True
    ops = {entry.name: set(operator_names((out / entry.model.path).read_bytes())) for entry in manifest.entries}
    assert "SQRT" in ops["fp32"]
    for name in ("fp32-npu", "int16x8-npu"):
        assert "SQRT" not in ops[name] and "MIRROR_PAD" not in ops[name]
    assert [(r.name, r.pair) for r in lowered.inputs] == [(r.name, r.pair) for r in semantic.inputs]
    assert [(r.name, r.pair) for r in lowered.outputs] == [(r.name, r.pair) for r in semantic.outputs]
    (prob,) = [i for i, r in enumerate(semantic.outputs) if r.pair is None]
    (hidden,) = [i for i, r in enumerate(semantic.outputs) if r.pair == 0]
    for index, lowered_atol, int16_atol in ((prob, 1e-3, 5e-3), (hidden, 0.02, 0.1)):
        want = outputs(out, semantic, f"output_{index}")
        assert want.std() > 0.005
        np.testing.assert_allclose(outputs(out, lowered, f"output_{index}"), want, atol=lowered_atol)
        record = int16.outputs[index]
        got = (outputs(out, int16, f"output_{index}").astype(np.float32) - record.zero_point) * record.scale
        np.testing.assert_allclose(got, want, atol=int16_atol)
    assert verify_manifest(out / "manifest.json").status == "ok"


@pytest.mark.skipif(importlib.util.find_spec("helia_model_zoo") is None, reason="needs helia-model-zoo")
def test_lowered_silero_goldens_pass_the_model_zoo_check(silero_npu):
    from helia_model_zoo import golden

    out, manifest = silero_npu
    for entry in manifest.entries:
        record = entry.reference.golden
        problems = golden.check(
            out / entry.model.path,
            out / record.file.path,
            kind=record.kind,
            steps=record.steps,
            resets=record.resets,
            resolver=record.resolver,
        )
        assert problems == [], (entry.name, problems)
