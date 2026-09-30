"""export_model on the TensorFlow backend: guards, records and identity with the legacy converter."""

import hashlib

import keras
import numpy as np
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)

from helia_edge.converters.tflite import ConversionType, QuantizationType, TfLiteKerasConverter  # noqa: E402
from helia_edge.export import VALID_IO, ExportSpec, IODType, Precision, TensorRole, export_model  # noqa: E402

LEGACY = {
    Precision.FP32: (QuantizationType.FP32, None, IODType.FLOAT32),
    Precision.FP32_FP16W: (QuantizationType.FP16, None, IODType.FLOAT32),
    Precision.FP16: (QuantizationType.FP16_NATIVE, None, IODType.FLOAT16),
    Precision.A8W8: (QuantizationType.INT8, "int8", IODType.INT8),
    Precision.A16W8: (QuantizationType.INT16X8, "int16", IODType.INT16),
}
MODES = {"keras": ConversionType.KERAS, "concrete": ConversionType.CONCRETE, "saved_model": ConversionType.SAVED_MODEL}


def build_model():
    keras.utils.set_random_seed(11)
    inputs = keras.Input((16, 16, 3), batch_size=1)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.DepthwiseConv2D(3, padding="same", dilation_rate=2)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    return keras.Model(inputs, keras.layers.Dense(4)(x))


@pytest.fixture(scope="module")
def model():
    return build_model()


@pytest.fixture(scope="module")
def calibration():
    return np.random.default_rng(0).standard_normal((8, 16, 16, 3)).astype(np.float32)


def legacy_convert(model, precision, mode, calibration):
    quantization, io_type, _ = LEGACY[precision]
    converter = TfLiteKerasConverter(model)
    try:
        test_x = calibration if precision in (Precision.A8W8, Precision.A16W8) else None
        return converter.convert(test_x, quantization=quantization, io_type=io_type, mode=MODES[mode])
    finally:
        converter.cleanup()


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("precision", list(Precision))
def test_export_model_matches_the_legacy_converter(model, calibration, precision, mode):
    spec = ExportSpec(precision=precision, io_dtype=LEGACY[precision][2], mode=mode)
    data = calibration if precision in (Precision.A8W8, Precision.A16W8) else None
    result = export_model(model, spec, data)
    assert result.content == legacy_convert(model, precision, mode, calibration)
    assert result.sha256 == hashlib.sha256(result.content).hexdigest()
    assert result.spec == spec


@pytest.mark.parametrize(("precision", "io_dtype"), [(p, d) for p in Precision for d in sorted(VALID_IO[p])])
def test_exported_io_dtype_is_the_requested_one(model, calibration, precision, io_dtype):
    data = calibration if precision in (Precision.A8W8, Precision.A16W8) else None
    result = export_model(model, ExportSpec(precision=precision, io_dtype=io_dtype, mode="concrete"), data)
    assert [r.dtype for r in (*result.inputs, *result.outputs)] == [io_dtype, io_dtype]


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [(QuantizationType.INT8, IODType.INT8), (QuantizationType.INT16X8, IODType.FLOAT32)],
)
def test_legacy_default_io_types(model, calibration, quantization, expected):
    from helia_edge.export.litert import tensor_records

    converter = TfLiteKerasConverter(model)
    try:
        content = converter.convert(calibration, quantization=quantization, mode=ConversionType.CONCRETE)
    finally:
        converter.cleanup()
    inputs, outputs = tensor_records(content)
    assert inputs[0].dtype == outputs[0].dtype == expected


@pytest.mark.parametrize("strict", [True, False])
def test_strict_reaches_the_converter(model, calibration, monkeypatch, strict):
    import tensorflow as tf

    from helia_edge.export import litert

    seen = []
    convert = litert.convert_litert

    def spy(*args, **kwargs):
        conversion = convert(*args, **kwargs)
        seen.append(list(conversion.converter.target_spec.supported_ops))
        return conversion

    monkeypatch.setattr(litert, "convert_litert", spy)
    export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete", strict=strict), calibration)
    assert (tf.lite.OpsSet.TFLITE_BUILTINS in seen[0]) is (not strict)


def test_records_describe_quantized_io(model, calibration):
    result = export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete"), calibration)
    (inp,), (out,) = result.inputs, result.outputs
    assert (inp.dtype, out.dtype) == (IODType.INT8, IODType.INT8)
    assert inp.role == out.role == TensorRole.SIGNAL
    assert inp.shape == (1, 16, 16, 3) and out.shape == (1, 4)
    assert inp.scale > 0 and out.scale > 0 and isinstance(inp.zero_point, int)


def test_records_describe_float_io(model):
    native = export_model(model, ExportSpec(precision="fp16", io_dtype="float16", mode="concrete"))
    assert [r.dtype for r in (*native.inputs, *native.outputs)] == [IODType.FLOAT16, IODType.FLOAT16]
    assert native.inputs[0].scale is None and native.inputs[0].zero_point is None
    spec = ExportSpec(precision="a8w8", io_dtype="float32", mode="concrete")
    float_io = export_model(model, spec, np.ones((2, 16, 16, 3), np.float32))
    assert float_io.inputs[0].dtype == IODType.FLOAT32


def test_dynamic_dimensions_calibrate_and_are_recorded():
    keras.utils.set_random_seed(2)
    inputs = keras.Input((None, 3), batch_size=1)
    model = keras.Model(inputs, keras.layers.Conv1D(4, 3, padding="same")(inputs))
    calibration = np.random.default_rng(3).standard_normal((4, 10, 3)).astype(np.float32)
    result = export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="keras"), calibration)
    assert result.inputs[0].shape == (1, -1, 3)


class FakeDistribution:
    def __init__(self, root, direct_url):
        self.version, self.root, self.direct_url = "9.9.9", root, direct_url

    def locate_file(self, name):
        return self.root / name

    def read_text(self, name):
        return self.direct_url if name == "direct_url.json" else None


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        ("vcs", ("9.9.9", "abc123")),
        ("editable", ("9.9.9", None)),
        ("wheel", ("9.9.9", None)),
        ("other-tree", ("unknown", None)),
    ],
)
def test_environment_identity_of_installed_distributions(monkeypatch, tmp_path, layout, expected):
    import importlib.metadata
    import json

    import helia_edge
    from helia_edge.export import result

    site, source = tmp_path / "site", tmp_path / "src"
    imported = source if layout == "editable" else site
    monkeypatch.setattr(helia_edge, "__file__", str(imported / "helia_edge" / "__init__.py"))
    direct_url = {
        "vcs": json.dumps({"url": "https://example.invalid/helia-edge.git", "vcs_info": {"commit_id": "abc123"}}),
        "editable": json.dumps({"url": f"file://{source}", "dir_info": {"editable": True}}),
        "wheel": None,
        "other-tree": None,
    }[layout]
    root = tmp_path / "elsewhere" if layout == "other-tree" else site
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: FakeDistribution(root, direct_url))
    env = result.environment_record()
    assert (env.helia_edge, env.helia_edge_commit) == expected


def test_environment_is_unknown_for_an_uninstalled_tree(monkeypatch, tmp_path):
    import helia_edge
    from helia_edge.export.result import environment_record

    monkeypatch.setattr(helia_edge, "__file__", str(tmp_path / "helia_edge" / "__init__.py"))
    env = environment_record()
    assert (env.helia_edge, env.helia_edge_commit) == ("unknown", None)


def test_environment_is_recorded(model):
    env = export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")).environment
    packages = dict(env.packages)
    assert env.helia_edge and env.python
    assert packages["tensorflow"] and packages["keras"] and packages["numpy"]


@pytest.mark.parametrize("precision", ["a8w8", "a16w8"])
def test_calibrated_precision_requires_calibration(model, precision):
    io = {"a8w8": "int8", "a16w8": "int16"}[precision]
    with pytest.raises(ValueError, match="requires calibration"):
        export_model(model, ExportSpec(precision=precision, io_dtype=io, mode="concrete"))


def test_calibration_is_rejected_for_float_precisions(model, calibration):
    with pytest.raises(ValueError, match="not calibrated"):
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete"), calibration)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros((4, 16, 16, 3), np.float64), "float32"),
        (np.zeros((0, 16, 16, 3), np.float32), "empty"),
        (np.zeros((4, 16, 16), np.float32), "does not match"),
        (np.zeros((4, 8, 16, 3), np.float32), "does not match"),
        (np.full((4, 16, 16, 3), np.nan, np.float32), "NaN or infinity"),
        (np.array(1.0, np.float32), "does not match"),
        ([[0.0]], "float32"),
    ],
)
def test_malformed_calibration_is_rejected(model, data, message):
    with pytest.raises(ValueError, match=message):
        export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete"), data)


def test_multi_input_models_are_refused():
    a, b = keras.Input((4,), batch_size=1), keras.Input((4,), batch_size=1)
    model = keras.Model([a, b], keras.layers.Add()([a, b]))
    with pytest.raises(ValueError, match="single-input"):
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete"))


def test_unknown_format_is_refused(model):
    with pytest.raises(ValueError, match="Unknown export format"):
        export_model(model, ExportSpec(format="onnx", precision="fp32", io_dtype="float32", mode="concrete"))


@pytest.mark.parametrize("quantization", [QuantizationType.INT8, QuantizationType.INT16X8])
def test_legacy_converter_refuses_calibrated_conversion_without_data(model, quantization):
    converter = TfLiteKerasConverter(model)
    try:
        with pytest.raises(ValueError, match="requires representative data"):
            converter.convert(quantization=quantization, mode=ConversionType.CONCRETE)
    finally:
        converter.cleanup()


def test_legacy_converter_keeps_permissive_io_type_and_defaults(model, calibration):
    converter = TfLiteKerasConverter(model)
    try:
        # io_type is ignored for float formats, as before; heartKIT passes "int8" for every format.
        fp32 = converter.convert(quantization=QuantizationType.FP32, io_type="int8", mode=ConversionType.CONCRETE)
        int16 = converter.convert(calibration, quantization=QuantizationType.INT16X8, mode=ConversionType.CONCRETE)
    finally:
        converter.cleanup()
    assert fp32 == export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")).content
    spec = ExportSpec(precision="a16w8", io_dtype="float32", mode="concrete")  # legacy INT16X8 default IO
    assert int16 == export_model(model, spec, calibration).content
