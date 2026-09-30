"""LiteRTRunner: run exported bytes on ai-edge-litert."""

import keras
import numpy as np
import pytest

from helia_edge.export import ExportSpec, LiteRTRunner, export_model

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)
pytest.importorskip("ai_edge_litert.interpreter")


@pytest.fixture(scope="module")
def model():
    keras.utils.set_random_seed(5)
    inputs = keras.Input((8, 8, 2), batch_size=1)
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    return keras.Model(inputs, keras.layers.Dense(3)(keras.layers.Flatten()(x)))


@pytest.fixture(scope="module")
def x():
    return np.random.default_rng(1).standard_normal((5, 8, 8, 2)).astype(np.float32)


def test_float_export_matches_keras(model, x):
    runner = LiteRTRunner(
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")).content
    )
    np.testing.assert_allclose(runner.predict(x), model.predict(x, verbose=0), atol=1e-5)


def test_int8_export_encodes_and_decodes(model, x):
    content = export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete"), x).content
    runner = LiteRTRunner(content)
    encoded = runner.encode(x)
    assert encoded.dtype == np.int8
    scale, zero_point = runner.input["quantization"]
    np.testing.assert_array_equal(encoded, np.clip(np.rint(x / scale + zero_point), -128, 127).astype(np.int8))
    y = runner.predict(x)
    assert y.dtype == np.float32 and y.shape == (5, 3)
    np.testing.assert_allclose(y, model.predict(x, verbose=0), atol=0.1)


def test_reference_kernels_agree_with_optimized_kernels_on_int8(model, x):
    content = export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete"), x).content
    reference, optimized = LiteRTRunner(content, reference_kernels=True), LiteRTRunner(content)
    np.testing.assert_array_equal(reference.run(reference.encode(x)), optimized.run(optimized.encode(x)))


def test_native_float16_runs_where_the_runtime_has_kernels(model, x):
    content = export_model(model, ExportSpec(precision="fp16", io_dtype="float16", mode="concrete")).content
    runner = LiteRTRunner(content)
    assert runner.encode(x).dtype == np.float16
    np.testing.assert_allclose(runner.predict(x), model.predict(x, verbose=0), atol=2e-2)


def test_run_refuses_the_wrong_input_dtype(model, x):
    runner = LiteRTRunner(
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")).content
    )
    with pytest.raises(ValueError, match="does not match"):
        runner.run(x.astype(np.float64))
