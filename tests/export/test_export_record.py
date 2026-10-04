"""export(): a static-batch LiteRT artifact with its export record; the record schema and load_export_record."""

import json
from pathlib import Path

import keras
import numpy as np
import pydantic
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)
pytest.importorskip("ai_edge_litert.interpreter")

from helia_edge.export import (  # noqa: E402
    RECORD_SCHEMA,
    ExportOptions,
    ExportRecord,
    ExportSpec,
    export,
    export_model,
    load_export_record,
    weights_digest,
)
from helia_edge.export.golden import golden_npz  # noqa: E402
from helia_edge.export.litert import operator_names  # noqa: E402
from helia_edge.export.record import Source, WeightImport  # noqa: E402
from helia_edge.models import ModelSpec, SileroVadParams, TcnParams, build, compact_tcn_params  # noqa: E402

KIT_SPECS = json.loads((Path(__file__).parents[1] / "fixtures" / "kit-tcn-specs.json").read_text())
"""TCN configurations of heartKIT PPG denoise, heartKIT PPG beat segmentation and sleepKIT apnea."""

SPEC = ModelSpec(params=compact_tcn_params(num_classes=3), input_shape=(64, 4))


def seeded(spec, seed=0, batch_size=None):
    keras.utils.set_random_seed(seed)
    return build(spec, batch_size=batch_size)


def samples(shape, count=8, seed=1):
    return np.random.default_rng(seed).normal(size=(count, *shape)).astype(np.float32)


@pytest.mark.parametrize("name", sorted(KIT_SPECS))
def test_kit_tcns_export_with_a_static_batch_and_no_space_to_batch(name):
    spec = ModelSpec(
        params=TcnParams.model_validate(KIT_SPECS[name]["params"]), input_shape=KIT_SPECS[name]["input_shape"]
    )
    model = seeded(spec)  # a dynamic batch, as the kits train
    assert model.inputs[0].shape[0] is None
    result = export(model, precision="a8w8", io_dtype="int8", calibration=samples(spec.input_shape), spec=spec)
    ops = operator_names(result.content)
    assert not {"SPACE_TO_BATCH_ND", "BATCH_TO_SPACE_ND"} & set(ops)
    assert result.record.export.batch_size == 1 and result.model.inputs[0].shape[0] == 1
    assert all(entry.shape[0] == 1 for entry in result.record.io.inputs)


def test_the_export_is_the_export_model_artifact_with_its_record():
    model = seeded(SPEC, batch_size=1)
    calibration = samples(SPEC.input_shape)
    result = export(model, precision="a8w8", io_dtype="int8", calibration=calibration, spec=SPEC)
    spec = ExportSpec(precision="a8w8", io_dtype="int8", mode="keras")
    assert result.content == export_model(model, spec, calibration).content
    record = result.record
    assert record.model == SPEC and record.weights.digest == weights_digest(model)
    assert record.export.calibration.samples == len(calibration) and record.export.options == ExportOptions()
    assert record.artifact.file == "model.tflite" and record.artifact.bytes == len(result.content)
    assert len(record.io.inputs) == len(model.inputs) and record.io.inputs[0].shape == (1, *SPEC.input_shape)
    assert record.environment.helia_edge.source in ("release", "vcs", "local", "unknown")
    again = export(model, precision="a8w8", io_dtype="int8", calibration=calibration, spec=SPEC)
    assert again.record == record


def test_a_dynamic_batch_needs_a_spec_and_a_fixed_batch_must_match():
    with pytest.raises(ValueError, match="batch"):
        export(seeded(SPEC), precision="fp32", io_dtype="float32")
    with pytest.raises(ValueError, match="batch"):
        export(seeded(SPEC, batch_size=2), precision="fp32", io_dtype="float32", spec=SPEC)
    rebuilt = export(seeded(SPEC), precision="fp32", io_dtype="float32", spec=SPEC, batch_size=2)
    assert rebuilt.model.inputs[0].shape[0] == 2 and rebuilt.record.export.batch_size == 2
    other = ModelSpec(params=compact_tcn_params(filters=16, num_classes=3), input_shape=(64, 4))
    with pytest.raises(ValueError, match="shapes of build"):
        export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", spec=other)
    for batch_size in (0, True, 1.0):
        with pytest.raises(pydantic.ValidationError, match="batch_size"):
            export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", batch_size=batch_size)


def test_the_record_refuses_what_it_does_not_describe():
    model = seeded(SPEC, batch_size=1)
    with pytest.raises(pydantic.ValidationError, match="precision"):
        export(model, precision="fp32-w16", io_dtype="float32")
    with pytest.raises(pydantic.ValidationError, match="mode"):
        ExportOptions(mode="saved_model")
    record = export(model, precision="fp32", io_dtype="float32").record
    assert record.model is None and record.export.calibration is None
    data = json.loads(record.model_dump_json(by_alias=True))
    assert data["schema"] == RECORD_SCHEMA
    with pytest.raises(pydantic.ValidationError):
        ExportRecord.model_validate({**data, "target": "anything"})
    with pytest.raises(pydantic.ValidationError):
        ExportRecord.model_validate({**data, "schema": "helia-edge/manifest@1"})


def test_the_weights_digest_follows_the_weights_not_the_file(tmp_path):
    model = seeded(SPEC, batch_size=1)
    digest = weights_digest(model)
    model.save_weights(tmp_path / "w.weights.h5")
    model.save(tmp_path / "m.keras")
    reloaded = seeded(SPEC, seed=5, batch_size=1)
    assert weights_digest(reloaded) != digest
    reloaded.load_weights(tmp_path / "w.weights.h5")
    assert weights_digest(reloaded) == digest == weights_digest(keras.saving.load_model(tmp_path / "m.keras"))
    assert weights_digest(build(SPEC, batch_size=1, name="renamed")) != digest  # other values
    dense = [keras.Sequential([keras.Input((2,)), keras.layers.Dense(2, name=name)]) for name in ("a", "b")]
    dense[1].set_weights(dense[0].get_weights())  # the same values under other weight paths
    assert weights_digest(dense[0]) != weights_digest(dense[1])
    changed = model.weights[-1]
    changed.assign(changed + 1e-6)
    assert weights_digest(model) != digest


def test_a_written_export_loads_back_and_refuses_other_weights(tmp_path):
    model = seeded(SPEC, batch_size=1)
    result = export(model, precision="fp32", io_dtype="float32", spec=SPEC)
    path = result.write(tmp_path / "out")
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.tflite", "model.weights.h5", "record.json"]
    assert ExportRecord.read(path) == result.record
    loaded = load_export_record(path)
    x = samples(SPEC.input_shape, count=1)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(loaded(x)), keras.ops.convert_to_numpy(model(x)))
    seeded(SPEC, seed=9, batch_size=1).save_weights(tmp_path / "other.weights.h5")
    with pytest.raises(ValueError, match="digest"):
        load_export_record(path, tmp_path / "other.weights.h5")
    unspecified = export(model, precision="fp32", io_dtype="float32").write(tmp_path / "bare")
    with pytest.raises(ValueError, match="no model spec"):
        load_export_record(unspecified)


def test_a_streaming_export_records_its_calibration_import_and_golden(tmp_path):
    spec = ModelSpec(params=SileroVadParams(stft="conv_blocks", magnitude="max_projection", encoder_tail="live_taps"))
    model = seeded(spec, batch_size=1)
    rng = np.random.default_rng(3)
    for weight in model.weights:  # scaled so every layer stays in a useful range
        shape = tuple(weight.shape)
        scale = 4.0 if "basis" in weight.path else (0.1 if len(shape) == 1 else 1 / np.sqrt(np.prod(shape[:-1])))
        weight.assign((scale * rng.standard_normal(shape)).astype(np.float32))
    t = np.arange(512 * 64 + 64) / 16000
    signal = 0.05 * rng.standard_normal(t.size) + 0.4 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    calls = np.stack([signal[i * 512 : i * 512 + 576] for i in range(64)]).astype(np.float32)
    imported = WeightImport(mapping="silero_vad_v6_onnx", source=Source(sha256="7" * 64))
    result = export(
        model, precision="a16w8", io_dtype="int16", calibration=calls, resets=(32,), spec=spec, weights_import=imported
    ).with_golden(calls[:16], resets=(8,))
    # Sqrt-free, mirror-pad-free
    assert not {"SQRT", "MIRROR_PAD", "TRANSPOSE"} & set(operator_names(result.content))
    record = result.record
    assert record.export.calibration.resets == (32,) and record.weights.import_ == imported
    assert record.io.state_scales_tied is True
    assert (
        record.golden.steps == 16
        and record.golden.resets == (8,)
        and record.golden.schema_ == "helia-model-zoo/golden@2"
    )
    path = result.write(tmp_path)
    assert (tmp_path / "golden.npz").stat().st_size == record.golden.file.bytes
    golden = np.load(tmp_path / "golden.npz")
    assert golden["input_0"].shape[0] == 16 and golden["input_0"].dtype == np.int16
    assert (tmp_path / "golden.npz").read_bytes() == golden_npz(result.content, calls[:16], (8,))
    assert golden_npz(result.content, calls[:16]) != result.golden  # the reset changes the sequence
    assert json.loads(path.read_text())["weights"]["import"]["mapping"] == "silero_vad_v6_onnx"
    with pytest.raises(ValueError, match="streaming"):
        export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", resets=(1,))
