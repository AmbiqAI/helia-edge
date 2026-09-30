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


def test_int8_encode_saturates(model, x):
    runner = LiteRTRunner(
        export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete"), x).content
    )
    encoded = runner.encode(np.stack([np.full((8, 8, 2), 1e6, np.float32), np.full((8, 8, 2), -1e6, np.float32)]))
    assert encoded[0].min() == 127 and encoded[1].max() == -128


def test_int16_export_runs(model, x):
    runner = LiteRTRunner(
        export_model(model, ExportSpec(precision="a16w8", io_dtype="int16", mode="concrete"), x).content
    )
    assert runner.encode(x).dtype == np.int16
    np.testing.assert_allclose(runner.predict(x), model.predict(x, verbose=0), atol=0.05)


def test_reference_kernels_agree_with_optimized_kernels_on_int8(model, x):
    content = export_model(model, ExportSpec(precision="a8w8", io_dtype="int8", mode="concrete"), x).content
    reference, optimized = LiteRTRunner(content, reference_kernels=True), LiteRTRunner(content)
    raw = [runner.run(runner.encode(x)).astype(np.int32) for runner in (reference, optimized)]
    # Kernel implementations may round requantization differently; they agree to one step.
    assert np.abs(raw[0] - raw[1]).max() <= 1


def test_native_float16_runs_where_the_runtime_has_kernels(model, x):
    content = export_model(model, ExportSpec(precision="fp16", io_dtype="float16", mode="concrete")).content
    runner = LiteRTRunner(content)
    assert runner.encode(x).dtype == np.float16
    np.testing.assert_allclose(runner.predict(x), model.predict(x, verbose=0), atol=2e-2)


def test_dynamic_dimensions_run_at_each_sample_size():
    keras.utils.set_random_seed(4)
    inputs = keras.Input((None, 3), batch_size=1)
    model = keras.Model(inputs, keras.layers.Conv1D(2, 3, padding="same")(inputs))
    runner = LiteRTRunner(export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="keras")).content)
    for length in (5, 9):
        x = np.random.default_rng(length).standard_normal((2, length, 3)).astype(np.float32)
        np.testing.assert_allclose(runner.predict(x), model.predict(x, verbose=0), atol=1e-5)


def test_fixed_dimensions_are_not_resized():
    # Convolution plus global pooling would run at another size if fixed dimensions were resized.
    inputs = keras.Input((8, 8, 2), batch_size=1)
    pooled = keras.layers.GlobalAveragePooling2D()(keras.layers.Conv2D(2, 3, padding="same")(inputs))
    spec = ExportSpec(precision="fp32", io_dtype="float32", mode="keras")
    runner = LiteRTRunner(export_model(keras.Model(inputs, pooled), spec).content)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(np.zeros((1, 6, 8, 2), np.float32))


def test_run_refuses_the_wrong_input_dtype(model, x):
    runner = LiteRTRunner(
        export_model(model, ExportSpec(precision="fp32", io_dtype="float32", mode="concrete")).content
    )
    with pytest.raises(ValueError, match="does not match"):
        runner.run(x.astype(np.float64))
