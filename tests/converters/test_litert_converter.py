import importlib
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

from helia_edge.converters import litert
from helia_edge.converters.litert import ConversionType, LiteRTKerasConverter, QuantizationType


def _make_model() -> keras.Model:
    inputs = keras.Input(shape=(2,), name="features")
    x = keras.layers.Dense(3, activation="relu", name="dense1")(inputs)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.set_weights(
        [
            np.array([[1.0, -1.0, 0.5], [0.25, 0.75, -0.5]], dtype=np.float32),
            np.array([0.1, -0.2, 0.3], dtype=np.float32),
            np.array([[1.0], [-0.5], [0.25]], dtype=np.float32),
            np.array([0.05], dtype=np.float32),
        ]
    )
    return model


@pytest.fixture()
def sample_x() -> np.ndarray:
    return np.array([[0.2, 0.4], [1.0, -0.5], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)


@pytest.fixture()
def sample_y(sample_x: np.ndarray) -> np.ndarray:
    return np.array([[0.4], [0.6], [0.3], [0.5]], dtype=np.float32)


def test_public_litert_imports():
    assert litert.LiteRTKerasConverter is LiteRTKerasConverter
    assert litert.ConversionType is ConversionType
    assert litert.QuantizationType is QuantizationType


def test_missing_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch, sample_x: np.ndarray):
    real_import_module = importlib.import_module

    def _raise(name: str):
        if name == "ai_edge_litert.interpreter":
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setattr("helia_edge.converters.litert.converter.importlib.import_module", _raise)

    converter = LiteRTKerasConverter(_make_model())
    with pytest.raises(ImportError, match="helia-edge\\[litert\\]"):
        converter.convert(sample_x)


@pytest.mark.parametrize("mode", [ConversionType.KERAS, ConversionType.SAVED_MODEL, ConversionType.CONCRETE])
def test_convert_returns_non_empty_flatbuffer(mode: ConversionType, sample_x: np.ndarray):
    converter = LiteRTKerasConverter(_make_model())
    content = converter.convert(sample_x, mode=mode)
    assert isinstance(content, bytes)
    assert content


@pytest.mark.parametrize("mode", [ConversionType.KERAS, ConversionType.CONCRETE])
def test_predict_matches_keras_for_signature_and_non_signature_modes(mode: ConversionType, sample_x: np.ndarray):
    pytest.importorskip("ai_edge_litert.interpreter")
    model = _make_model()
    converter = LiteRTKerasConverter(model)
    converter.convert(sample_x, mode=mode)

    expected = model(sample_x, training=False).numpy()
    actual = converter.predict(sample_x)

    assert np.allclose(actual, expected, atol=1e-3)


def test_export_writes_bytes_and_litert_interpreter_loads_file(tmp_path: Path, sample_x: np.ndarray):
    interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter
    converter = LiteRTKerasConverter(_make_model())
    content = converter.convert(sample_x)
    model_path = tmp_path / "model.litert"

    converter.export(model_path)

    assert model_path.read_bytes() == content
    runtime = interpreter(model_path=str(model_path))
    runtime.allocate_tensors()
    assert runtime.get_input_details()


def test_export_header_writes_c_header(tmp_path: Path, sample_x: np.ndarray):
    converter = LiteRTKerasConverter(_make_model())
    converter.convert(sample_x)
    header_path = tmp_path / "model.h"

    converter.export_header(header_path, name="litert_model")

    header = header_path.read_text(encoding="utf-8")
    assert "const unsigned char litert_model[] = {" in header
    assert "const unsigned int litert_model_len =" in header
    assert "#ifndef __LITERT_MODEL_H" in header


def test_evaluate_returns_valid_loss(sample_x: np.ndarray, sample_y: np.ndarray):
    model = _make_model()
    converter = LiteRTKerasConverter(model)
    converter.convert(sample_x)

    loss = converter.evaluate(sample_x, sample_y)
    expected = keras.losses.get(model.loss)(sample_y, model(sample_x, training=False).numpy()).numpy()

    assert np.all(np.isfinite(loss))
    assert np.allclose(loss, expected, atol=1e-3)


def test_export_methods_require_convert(tmp_path: Path):
    converter = LiteRTKerasConverter(_make_model())

    with pytest.raises(ValueError, match="convert\\(\\) first"):
        converter.export(tmp_path / "model.litert")

    with pytest.raises(ValueError, match="convert\\(\\) first"):
        converter.export_header(tmp_path / "model.h")


def test_saved_model_cleanup_removes_temporary_export_dir(sample_x: np.ndarray):
    converter = LiteRTKerasConverter(_make_model())

    converter.convert(sample_x, mode=ConversionType.SAVED_MODEL)
    temp_dir = Path(converter.tf_model_path.name)

    assert temp_dir.exists()
    converter.cleanup()
    assert not temp_dir.exists()


def test_invalid_mode_and_missing_representative_data_fail_explicitly():
    converter = LiteRTKerasConverter(_make_model())

    with pytest.raises(ValueError, match="INVALID"):
        converter.convert(mode="INVALID")

    with pytest.raises(ValueError, match="representative data"):
        converter.convert(quantization=QuantizationType.INT8)


def test_int8_conversion_runs_with_calibration_data(sample_x: np.ndarray):
    pytest.importorskip("ai_edge_litert.interpreter")
    converter = LiteRTKerasConverter(_make_model())

    content = converter.convert(sample_x, quantization=QuantizationType.INT8, io_type="int8")

    assert content


def test_quantized_strict_false_appends_builtin_fallback(sample_x: np.ndarray, monkeypatch: pytest.MonkeyPatch):
    converter = LiteRTKerasConverter(_make_model())
    fallback_converter = type(
        "FakeConverter",
        (),
        {
            "optimizations": [],
            "target_spec": type("TargetSpec", (), {"supported_ops": [], "supported_types": []})(),
            "inference_input_type": None,
            "inference_output_type": None,
            "representative_dataset": None,
            "convert": lambda self: b"litert",
        },
    )()

    monkeypatch.setattr("helia_edge.converters.litert.converter._load_litert_interpreter", lambda: object)
    monkeypatch.setattr(tf.lite.TFLiteConverter, "from_keras_model", lambda **kwargs: fallback_converter)

    converter.convert(sample_x, quantization=QuantizationType.INT8, strict=False)

    assert tf.lite.OpsSet.TFLITE_BUILTINS_INT8 in fallback_converter.target_spec.supported_ops
    assert tf.lite.OpsSet.TFLITE_BUILTINS in fallback_converter.target_spec.supported_ops
