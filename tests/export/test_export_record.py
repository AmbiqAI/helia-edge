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
    Source,
    WeightImport,
    export,
    export_model,
    load_export_record,
    weights_digest,
)
from helia_edge.export.golden import golden_npz  # noqa: E402
from helia_edge.export.litert import operator_names  # noqa: E402
from helia_edge.models import (  # noqa: E402
    ModelSpec,
    SileroVadParams,
    TcnParams,
    TsMixerParams,
    build,
    compact_tcn_params,
)

KIT_SPECS = json.loads((Path(__file__).parents[1] / "fixtures" / "kit-tcn-specs.json").read_text())
"""TCN configurations of heartKIT PPG denoise, heartKIT PPG beat segmentation and sleepKIT apnea."""

SPEC = ModelSpec(params=compact_tcn_params(num_classes=3), input_shape=(64, 4))


class Holder(keras.layers.Layer):
    """A layer holding one weight of the given shape and dtype; ``weights_digest`` reads ``.weights``."""

    def __init__(self, shape, dtype, **kwargs):
        super().__init__(**kwargs)
        self.value = self.add_weight(shape=shape, dtype=dtype, initializer="zeros", trainable=False)


def npy_sha256(array):
    import hashlib
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


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
    assert result.content == export_model(result.model, spec, calibration).content
    bare = export(model, precision="a8w8", io_dtype="int8", calibration=calibration)  # no spec: the model as is
    assert bare.content == export_model(model, spec, calibration).content and bare.model is model
    record = result.record
    assert record.model == SPEC and record.weights.digest == weights_digest(model)
    assert record.export.calibration.samples == len(calibration) and record.export.options == ExportOptions()
    assert record.export.calibration.sha256 == npy_sha256(calibration)
    from_float64 = export(
        model, precision="a8w8", io_dtype="int8", calibration=calibration.astype(np.float64), spec=SPEC
    )
    assert from_float64.content == result.content and from_float64.record == record  # cast to float32, then hashed
    import ml_dtypes

    for other in (calibration.astype(ml_dtypes.bfloat16), np.round(calibration * 100).astype(np.int16)):
        expected = export(model, precision="a8w8", io_dtype="int8", calibration=other.astype(np.float32), spec=SPEC)
        assert export(model, precision="a8w8", io_dtype="int8", calibration=other, spec=SPEC).record == expected.record
    for wrong in (
        calibration.astype(np.complex64),
        calibration.astype(str),
        calibration > 0,
        np.float32(1.0),
        {"x": 1},
    ):
        with pytest.raises(ValueError, match="float or integer array"):
            export(model, precision="a8w8", io_dtype="int8", calibration=wrong, spec=SPEC)
    assert record.artifact.file == "model.tflite" and record.artifact.bytes == len(result.content)
    assert len(record.io.inputs) == len(model.inputs) and record.io.inputs[0].shape == (1, *SPEC.input_shape)
    assert record.environment.helia_edge.source in ("release", "vcs", "local", "unknown")
    again = export(model, precision="a8w8", io_dtype="int8", calibration=calibration, spec=SPEC)
    assert again.record == record


def test_without_a_spec_the_batch_must_match_and_with_one_the_spec_is_built():
    with pytest.raises(ValueError, match="Pass spec to export"):  # refused before converting
        export(seeded(SPEC), precision="fp32", io_dtype="float32")
    with pytest.raises(ValueError, match="Pass spec to export"):
        export(seeded(SPEC, batch_size=2), precision="fp32", io_dtype="float32")
    assert (
        export(seeded(SPEC, batch_size=2), precision="fp32", io_dtype="float32", spec=SPEC).record.export.batch_size
        == 1
    )
    rebuilt = export(seeded(SPEC), precision="fp32", io_dtype="float32", spec=SPEC, batch_size=2)
    assert rebuilt.model.inputs[0].shape[0] == 2 and rebuilt.record.export.batch_size == 2
    other = ModelSpec(params=compact_tcn_params(filters=16, num_classes=3), input_shape=(64, 4))
    with pytest.raises(ValueError, match="weights do not have the shapes of build"):
        export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", spec=other)
    longer = SPEC.model_copy(update={"input_shape": (128, 4)})  # the same weights for a longer input
    with pytest.raises(ValueError, match="inputs do not have the shapes of build"):
        export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", spec=longer)
    for precision, io_dtype in (("a8w8", "int8"), ("a16w8", "int16")):
        with pytest.raises(ValueError, match="batch_size 1"):
            calibration = samples(SPEC.input_shape)
            export(
                seeded(SPEC), precision=precision, io_dtype=io_dtype, calibration=calibration, spec=SPEC, batch_size=2
            )
    for batch_size in (0, True, 1.0):
        with pytest.raises(pydantic.ValidationError, match="batch_size"):
            export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", batch_size=batch_size)


def test_with_a_spec_the_export_is_the_spec_with_the_model_weights():
    params = SPEC.params.model_dump()
    for block in params["blocks"]:
        block["dilation"], block["activation"] = [1, 1], "relu"  # same weight shapes, another graph
    other = seeded(ModelSpec(params=TcnParams.model_validate(params), input_shape=SPEC.input_shape), batch_size=1)
    result = export(other, precision="fp32", io_dtype="float32", spec=SPEC)
    expected = seeded(SPEC, batch_size=1)
    expected.set_weights(other.get_weights())
    x = samples(SPEC.input_shape, count=1)
    got = keras.ops.convert_to_numpy(result.model(x))
    np.testing.assert_array_equal(got, keras.ops.convert_to_numpy(expected(x)))
    assert not np.array_equal(got, keras.ops.convert_to_numpy(other(x)))


def test_export_leaves_the_callers_keras_state_alone():
    model = seeded(SPEC, batch_size=1)
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    floatx = keras.config.floatx()
    try:
        keras.config.set_dtype_policy("float32")
        default = export(model, precision="fp32", io_dtype="float32", spec=SPEC).content
        keras.config.set_dtype_policy("mixed_float16")
        keras.config.set_floatx("float16")
        before = keras.layers.Dense(2, dtype="float32").name
        assert export(model, precision="fp32", io_dtype="float32", spec=SPEC).content == default
        with pytest.raises(ValueError, match="shapes of build"):
            export(model, precision="fp32", io_dtype="float32", spec=SPEC.model_copy(update={"input_shape": (128, 4)}))
        assert keras.config.dtype_policy().name == "mixed_float16" and keras.config.floatx() == "float16"
        index = int(before.rsplit("_", 1)[1]) if "_" in before else 0
        assert keras.layers.Dense(2, dtype="float32").name == f"dense_{index + 1}"  # the caller's numbering goes on
    finally:
        keras.config.set_dtype_policy(previous)
        keras.config.set_floatx(floatx)


def test_the_reference_build_restores_the_callers_state_when_it_fails():
    from helia_edge.export.api import _reference_build

    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_bfloat16")
    try:
        before = keras.layers.Dense(2, dtype="float32").name
        with pytest.raises(RuntimeError), _reference_build():
            assert keras.config.dtype_policy().name == "float32"
            raise RuntimeError("build failed")
        assert keras.config.dtype_policy().name == "mixed_bfloat16"
        index = int(before.rsplit("_", 1)[1]) if "_" in before else 0
        assert keras.layers.Dense(2, dtype="float32").name == f"dense_{index + 1}"
    finally:
        keras.config.set_dtype_policy(previous)


@pytest.mark.parametrize("table", ["present", "moved", "unimportable"])
def test_the_reference_build_warns_when_keras_numbers_names_elsewhere(monkeypatch, table):
    """Layer numbering is reset through Keras's private name table; a Keras that moved it gets a warning."""
    import collections
    import sys
    import types
    import warnings

    import keras.src.backend.common as common
    from keras.src.utils import naming

    from helia_edge.export.api import _reference_build

    if table == "moved":
        elsewhere = collections.defaultdict(int)
        monkeypatch.setattr(
            naming, "global_state", types.SimpleNamespace(get_global_attribute=lambda *a, **k: elsewhere)
        )
    elif table == "unimportable":
        monkeypatch.delattr(common, "global_state")
        monkeypatch.setitem(sys.modules, "keras.src.backend.common.global_state", None)
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("mixed_float16")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _reference_build():
                assert keras.config.dtype_policy().name == "float32"
                if table == "present":  # the probe layer leaves no name behind
                    assert keras.layers.Identity().name == "identity"
        assert keras.config.dtype_policy().name == "mixed_float16"
    finally:
        keras.config.set_dtype_policy(previous)
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    if table == "present":
        assert messages == []
    else:
        assert len(messages) == 1 and "may depend on layers built earlier" in messages[0]


def test_a_record_loads_under_any_caller_policy(tmp_path):
    model = seeded(SPEC, batch_size=1)
    path = export(model, precision="fp32", io_dtype="float32", spec=SPEC).write(tmp_path)
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("float16")
    try:
        loaded = load_export_record(path)
    finally:
        keras.config.set_dtype_policy(previous)
    x = samples(SPEC.input_shape, count=1)
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(loaded(x)), keras.ops.convert_to_numpy(model(x)))


def test_the_record_refuses_what_it_does_not_describe():
    model = seeded(SPEC, batch_size=1)
    with pytest.raises(pydantic.ValidationError, match="precision"):
        export(model, precision="fp32-w16", io_dtype="float32")
    with pytest.raises(pydantic.ValidationError, match="mode"):
        ExportOptions(mode="saved_model")
    with pytest.raises(pydantic.ValidationError, match="strict"):
        ExportOptions(strict="no")
    with pytest.raises(pydantic.ValidationError, match="state_tie_tolerance"):
        ExportOptions(state_tie_tolerance="0.1")
    concrete = ExportOptions(mode="concrete")
    with pytest.raises(pydantic.ValidationError, match="concrete mode traces batch 1"):
        export(seeded(SPEC), precision="fp32", io_dtype="float32", spec=SPEC, batch_size=2, options=concrete)
    at_one = export(seeded(SPEC), precision="fp32", io_dtype="float32", spec=SPEC, options=concrete).record
    assert at_one.io.inputs[0].shape[0] == at_one.export.batch_size == 1
    outputs = [
        {**e, "shape": [5, *e["shape"][1:]]} for e in json.loads(at_one.model_dump_json(by_alias=True))["io"]["outputs"]
    ]
    with pytest.raises(pydantic.ValidationError, match="output batch"):
        ExportRecord.model_validate(
            {
                **json.loads(at_one.model_dump_json(by_alias=True)),
                "io": {**json.loads(at_one.model_dump_json(by_alias=True))["io"], "outputs": outputs},
            }
        )
    record = export(model, precision="fp32", io_dtype="float32").record
    assert record.model is None and record.export.calibration is None
    data = json.loads(record.model_dump_json(by_alias=True))
    assert data["schema"] == RECORD_SCHEMA
    with pytest.raises(pydantic.ValidationError):
        ExportRecord.model_validate({**data, "target": "anything"})
    with pytest.raises(pydantic.ValidationError):
        ExportRecord.model_validate({**data, "schema": "helia-edge/manifest@1"})
    settings = data["export"]
    calibration = {"sha256": "0" * 64, "samples": 4, "resets": []}
    for wrong, message in (
        ({**settings, "precision": "fp16", "io_dtype": "int8"}, "not valid for precision"),
        ({**settings, "precision": "a8w8", "io_dtype": "int8"}, "needs calibration"),
        ({**settings, "calibration": calibration}, "takes no calibration"),
        (
            {**settings, "precision": "a8w8", "io_dtype": "int8", "calibration": {**calibration, "resets": [4]}},
            "resets",
        ),
        (
            {**settings, "precision": "a16w8", "io_dtype": "int16", "calibration": calibration, "batch_size": 2},
            "batch_size 1",
        ),
        ({**settings, "batch_size": 2}, "output batch"),
    ):
        with pytest.raises(pydantic.ValidationError, match=message):
            ExportRecord.model_validate({**data, "export": wrong})


def test_the_weights_digest_is_the_documented_encoding():
    import hashlib

    model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(3)])
    kernel, bias = (np.arange(6, dtype=np.float32).reshape(2, 3), np.array([-1.0, 0.5, 2.0], np.float32))
    model.set_weights([kernel, bias])
    expected = hashlib.sha256()
    for value in (kernel, bias):  # model.weights order: kernel, then bias
        expected.update(b"float32\0" + ",".join(map(str, value.shape)).encode() + b"\0" + value.tobytes())
    assert weights_digest(model) == f"sha256:{expected.hexdigest()}"


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
    dense[1].set_weights(dense[0].get_weights())  # weight paths are not hashed: Keras numbers unnamed layers
    assert weights_digest(dense[0]) == weights_digest(dense[1])
    swapped = keras.Sequential([keras.Input((2,)), keras.layers.Dense(2), keras.layers.Dense(2)])
    kernel_a, bias_a, kernel_b, bias_b = swapped.get_weights()
    digest_before = weights_digest(swapped)
    swapped.set_weights([kernel_b, bias_a, kernel_a, bias_b])
    assert weights_digest(swapped) != digest_before  # the order of the weights is hashed
    as_int, as_float = (Holder((2,), dtype) for dtype in ("int32", "float32"))
    as_int.value.assign(np.arange(2, dtype=np.float32).view(np.int32))  # the same bytes as another dtype
    as_float.value.assign(np.arange(2, dtype=np.float32))
    assert weights_digest(as_int) != weights_digest(as_float)
    assert weights_digest(Holder((), "float32")) != weights_digest(Holder((1,), "float32"))  # zeros, other shapes
    wide, tall = (keras.Sequential([keras.Input((n,)), keras.layers.Dense(6 // n, use_bias=False)]) for n in (2, 3))
    tall.set_weights([wide.get_weights()[0].reshape(3, 2)])  # the same bytes in another shape
    assert weights_digest(wide) != weights_digest(tall)
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
    with pytest.raises(ValueError, match="for streaming models"):
        result.with_golden(samples(SPEC.input_shape, count=4))
    unspecified = export(model, precision="fp32", io_dtype="float32").write(tmp_path / "bare")
    with pytest.raises(ValueError, match="no model spec"):
        load_export_record(unspecified)


def test_a_record_reproduces_its_artifact_in_a_new_process(tmp_path):
    """TsMixer leaves layers unnamed, so Keras numbers them by how many it built in the process; neither the
    digest nor the artifact may depend on that."""
    import subprocess
    import sys

    spec = ModelSpec(params=TsMixerParams(blocks=[{"ff_dim": 8}], num_classes=3), input_shape=(16, 4))
    build(spec)  # numbering moves on
    path = export(seeded(spec), precision="fp32", io_dtype="float32", spec=spec).write(tmp_path)
    load_export_record(path)
    source = f"""
from helia_edge.export import ExportRecord, export, load_export_record
record = ExportRecord.read({str(path)!r})
again = export(load_export_record({str(path)!r}), precision="fp32", io_dtype="float32", spec=record.model)
assert again.record.artifact == record.artifact and again.record.weights == record.weights
"""
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def test_a_streaming_export_records_its_calibration_import_and_golden(tmp_path, monkeypatch):
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
    from helia_edge.export import api

    seen = []
    original = api.stream_calibration

    def spy(model, signals, resets=()):
        seen.append(tuple(resets))
        return original(model, signals, resets)

    monkeypatch.setattr(api, "stream_calibration", spy)
    result = export(
        model, precision="a16w8", io_dtype="int16", calibration=calls, resets=(32,), spec=spec, weights_import=imported
    ).with_golden(calls[:16], resets=(8,), uri="https://example.com/calls.npy")
    assert seen == [(32,)]  # the resets reach the state calibration
    # Sqrt-free, mirror-pad-free
    assert not {"SQRT", "MIRROR_PAD", "TRANSPOSE"} & set(operator_names(result.content))
    record = result.record
    assert record.export.calibration.resets == (32,) and record.weights.import_ == imported
    with pytest.raises(pydantic.ValidationError, match="resets"):
        record.golden.model_validate({**record.golden.model_dump(by_alias=True), "resets": (16,)})
    from_array = export(model, precision="a16w8", io_dtype="int16", calibration=calls, resets=np.array([32]), spec=spec)
    for resets in ([4.5], ["3"]):
        with pytest.raises(ValueError, match="integer steps"):
            export(model, precision="a16w8", io_dtype="int16", calibration=calls, resets=resets, spec=spec)
    assert from_array.record.export.calibration.resets == (32,)
    assert record.io.state_scales_tied is True
    assert record.golden.inputs == Source(sha256=npy_sha256(calls[:16]), uri="https://example.com/calls.npy")
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
    with pytest.raises(ValueError, match="no calibration"):
        export(seeded(SPEC, batch_size=1), precision="fp32", io_dtype="float32", resets=(1,))
    calibration = samples(SPEC.input_shape)
    with pytest.raises(ValueError, match="streaming models only"):
        export(seeded(SPEC, batch_size=1), precision="a8w8", io_dtype="int8", calibration=calibration, resets=(1,))
    for resets in ((0,), (-3, 3), (3, 3), (64,)):
        with pytest.raises(ValueError, match="resets"):
            export(model, precision="a16w8", io_dtype="int16", calibration=calls, resets=resets, spec=spec)
    with pytest.raises(ValueError, match="resets"):
        result.with_golden(calls[:4], resets=(4,))
    for resets in ([2.0], ["3"], 3, "3", b"\x08", bytearray(b"\x08"), [True]):
        with pytest.raises(ValueError, match="integer steps"):
            result.with_golden(calls[:4], resets=resets)
    with pytest.raises(ValueError, match="integer steps"):
        export(model, precision="a16w8", io_dtype="int16", calibration=calls, resets=3, spec=spec)
    with pytest.raises(ValueError, match="at least one call"):
        result.with_golden(calls[:0])
    with pytest.raises(ValueError, match="finite"):
        result.with_golden(np.full_like(calls[:2], np.nan))
    with pytest.raises(ValueError, match="float or integer array"):
        result.with_golden(calls[:2].astype(np.complex64))
    for unordered in ({1, 2}, {1: 0}):
        with pytest.raises(ValueError, match="ordered collection"):
            result.with_golden(calls[:4], resets=unordered)
    bare = export(model, precision="a16w8", io_dtype="int16", calibration=calls, resets=(32,), spec=spec)
    bare.write(tmp_path)
    assert not (tmp_path / "golden.npz").exists()  # a golden of an earlier write is removed
