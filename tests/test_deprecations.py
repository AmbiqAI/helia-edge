"""Legacy conversion and inference classes warn and point to helia_edge.export."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
pytest.importorskip("tensorflow")
if keras.backend.backend() != "tensorflow":
    pytest.skip("The legacy converters and interpreter need the TensorFlow backend", allow_module_level=True)

from helia_edge.converters.litert import LiteRTKerasConverter  # noqa: E402
from helia_edge.converters.tflite import QuantizationType, TfLiteKerasConverter  # noqa: E402
from helia_edge.interpreters.tflite import TfLiteKerasInterpreter  # noqa: E402


def small_model():
    return keras.Sequential([keras.Input((4,)), keras.layers.Dense(2)])


@pytest.mark.parametrize("cls", [TfLiteKerasConverter, LiteRTKerasConverter])
def test_converters_warn_and_name_the_replacement(cls):
    with pytest.warns(DeprecationWarning, match=rf"{cls.__name__} is deprecated; use helia_edge\.export\.export_model"):
        cls(small_model())


def test_the_interpreter_warns_and_still_predicts():
    model = small_model()
    with pytest.warns(DeprecationWarning):
        content = TfLiteKerasConverter(model).convert(quantization=QuantizationType.FP32, io_type="float32")
    with pytest.warns(
        DeprecationWarning, match=r"TfLiteKerasInterpreter is deprecated; use helia_edge\.export\.LiteRTRunner"
    ):
        interpreter = TfLiteKerasInterpreter(model_content=content)
    interpreter.compile()
    x = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    np.testing.assert_allclose(interpreter.predict(x), model.predict(x, verbose=0), atol=1e-5)


def test_from_saved_model_warns_once_at_the_caller(tmp_path):
    path = tmp_path / "model.keras"
    small_model().save(path)
    with pytest.warns(DeprecationWarning) as record:
        TfLiteKerasConverter.from_saved_model(path)
    deprecations = [
        w for w in record if issubclass(w.category, DeprecationWarning) and "is deprecated" in str(w.message)
    ]
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__
    with pytest.warns(DeprecationWarning, match="TfLiteKerasConverter is deprecated"):
        TfLiteKerasConverter(small_model())
