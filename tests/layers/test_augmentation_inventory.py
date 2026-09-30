"""Qualified existing transform inventory, config roundtrips and numeric guards."""

import json

import keras
import numpy as np
import pytest

import helia_edge.layers.preprocessing as pp

CASES = [
    ("Normalization2D", dict(mean=1.0, variance=4.0), (2, 6, 8, 2)),
    ("Rescaling1D", dict(scale=2.0), (2, 16, 2)),
    ("Rescaling2D", dict(scale=2.0), (2, 6, 8, 2)),
    ("LayerNormalization1D", dict(epsilon=0.01), (2, 16, 2)),
    ("LayerNormalization2D", dict(epsilon=0.01), (2, 6, 8, 2)),
    ("Resizing1D", dict(duration=8), (2, 16, 2)),
    ("Resizing2D", dict(height=3, width=4, interpolation="bilinear"), (2, 6, 8, 2)),
    ("RandomCrop1D", dict(duration=8), (2, 16, 2)),
    ("RandomCrop2D", dict(height=3, width=4), (2, 6, 8, 2)),
    ("RandomFlip2D", {}, (2, 6, 8, 2)),
    ("RandomChannel", dict(batchwise=True), (2, 6, 8, 2)),
    ("RandomChannel", dict(batchwise=False), (2, 6, 8, 2)),
    ("RandomCutout1D", dict(factor=(0.25, 0.25), cutouts=1), (2, 16, 2)),
    ("RandomCutout2D", dict(factor=(0.25, 0.25), cutouts=1), (2, 8, 8, 2)),
    ("RandomBackgroundNoises1D", dict(noises=np.arange(64, dtype="float32").reshape(32, 2), num_noises=2), (2, 16, 2)),
    ("RandomSineWave", dict(sample_rate=8, frequency=(1.0, 1.0), amplitude=(0.1, 0.1)), (2, 16, 2)),
    ("AddSineWave", dict(sample_rate=8, frequency=1.0, amplitude=0.1), (2, 16, 2)),
    ("RandomNoiseDistortion1D", dict(sample_rate=8, frequency=(1.0, 1.0), amplitude=(0.1, 0.1)), (2, 16, 2)),
    ("AmplitudeWarp", dict(sample_rate=8, frequency=(1.0, 1.0), amplitude=(0.9, 1.1)), (2, 16, 2)),
    ("FrequencyMixStyle2D", dict(probability=1.0, alpha=1.0, epsilon=0.01), (2, 6, 8, 2)),
    ("SpecAugment2D", dict(freq_mask_param=2, time_mask_param=3), (2, 6, 8, 2)),
    ("CascadedBiquadFilter", dict(sos=[[0.5, 0.25, 0.125, 1.0, -0.1, 0.05]]), (2, 16, 2)),
]


def array(x):
    return keras.ops.convert_to_numpy(x)


@pytest.mark.parametrize("name,kwargs,shape", CASES)
@pytest.mark.parametrize("channels_first", [False, True])
def test_concrete_config_and_execution(name, kwargs, shape, channels_first):
    x = np.arange(np.prod(shape), dtype="float32").reshape(shape) / 100
    if channels_first:
        x = np.moveaxis(x, -1, 1)
    layer = getattr(pp, name)(**kwargs, seed=42, data_format="channels_first" if channels_first else "channels_last")
    y = layer(x, training=True)
    assert np.isfinite(array(y)).all()
    config = keras.saving.serialize_keras_object(layer)
    json.dumps(config)
    restored = keras.saving.deserialize_keras_object(config)
    np.testing.assert_allclose(
        array(restored(x, training=False)), array(layer(x, training=False)), atol=1e-6, rtol=1e-6
    )
    if layer.training_only:
        np.testing.assert_array_equal(array(layer(x)), x)
        expected = getattr(pp, name)(**kwargs, seed=42, data_format=layer.data_format)(x, training=True)
        np.testing.assert_allclose(array(restored(x, training=True)), array(expected), atol=1e-6, rtol=1e-6)


def test_nearest_targets_preserve_large_integer_values():
    x = keras.ops.ones((1, 4, 1))
    y = np.array([2**40 + n for n in range(4)], dtype="int64").reshape(1, 4, 1)
    result = pp.Resizing1D(2, target_interpolation="nearest")(
        {
            "signals": {"x": x},
            "targets": {"ids": keras.ops.convert_to_tensor(y)},
            "masks": {"valid": keras.ops.convert_to_tensor(y % 2 == 0)},
        }
    )
    np.testing.assert_array_equal(array(result["targets"]["ids"]), y[:, [1, 3], :])
    np.testing.assert_array_equal(array(result["masks"]["valid"]), y[:, [1, 3], :] % 2 == 0)


def test_cutout_exact_count_zero_and_one():
    x = keras.ops.ones((2, 8, 1))
    np.testing.assert_array_equal(array(pp.RandomCutout1D(cutouts=0)(x, training=True)), array(x))
    y = array(pp.RandomCutout1D(factor=(0.25, 0.25), cutouts=1, fill_value=0.0, seed=42)(x, training=True))
    np.testing.assert_array_equal(np.sum(y == 0, axis=1), np.full((2, 1), 2))


def test_fixed_sine_reference():
    x = np.ones((2, 16, 2), dtype="float32")
    actual = array(pp.AddSineWave(sample_rate=8, frequency=1, amplitude=0.1)(x, training=False))
    expected = x + 0.1 * np.sin(2 * np.pi * np.arange(16) / 8)[None, :, None]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("forward_backward", [False, True])
def test_biquad_independent_recurrence(forward_backward):
    sos = np.array([[0.5, 0.25, 0.125, 1.0, -0.1, 0.05], [0.3, 0.1, 0.0, 1.0, -0.2, 0.0]], dtype="float64")
    x = np.sin(np.arange(64).reshape(2, 16, 2)).astype("float32")

    def reference(x):
        y = x.astype("float64").copy()
        for b0, b1, b2, a0, a1, a2 in sos:
            z = np.zeros_like(y)
            for t in range(y.shape[1]):
                z[:, t] = b0 * y[:, t]
                if t >= 1:
                    z[:, t] += b1 * y[:, t - 1] - a1 * z[:, t - 1]
                if t >= 2:
                    z[:, t] += b2 * y[:, t - 2] - a2 * z[:, t - 2]
            y = z / a0
        return y

    expected = reference(x)
    if forward_backward:
        expected = reference(expected[:, ::-1])[:, ::-1]
    actual = array(pp.CascadedBiquadFilter(sos=sos.tolist(), forward_backward=forward_backward)(x))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "name", ["AugmentationPipeline", "RandomChoice", "RandomAugmentation1DPipeline", "RandomAugmentation2DPipeline"]
)
def test_pipeline_serialization_and_branch_capture(tmp_path, name):
    layers = [pp.Rescaling1D(2.0), pp.Rescaling1D(3.0)]
    kwargs = {} if name == "AugmentationPipeline" else {"seed": 42}
    pipeline = getattr(pp, name)(layers=layers, **kwargs)
    x = keras.ops.ones((2, 8, 1))
    if name == "RandomChoice":
        for index, scale in ((0, 2.0), (1, 3.0)):
            np.testing.assert_array_equal(
                array(
                    pipeline(
                        x, training=True, transformations={"index": keras.ops.convert_to_tensor(index, dtype="int32")}
                    )
                ),
                np.full((2, 8, 1), scale),
            )
    y = pipeline(x, training=True)
    assert np.isfinite(array(y)).all()
    config = keras.saving.serialize_keras_object(pipeline)
    json.dumps(config)
    restored = keras.saving.deserialize_keras_object(config)
    np.testing.assert_array_equal(array(restored(x, training=False)), array(pipeline(x, training=False)))
    inputs = keras.Input(shape=(8, 1))
    model = keras.Model(inputs, pipeline(inputs, training=False))
    model.save(tmp_path / "pipeline.keras")
    reloaded = keras.models.load_model(tmp_path / "pipeline.keras")
    np.testing.assert_array_equal(array(reloaded(x)), array(model(x)))


def test_choice_builds_weighted_children_before_inference_save(tmp_path):
    children = [
        keras.layers.Dense(1, use_bias=False, kernel_initializer=keras.initializers.Constant(v)) for v in (2.0, 3.0)
    ]
    layer = pp.RandomChoice(children, seed=42)
    inputs = keras.Input(shape=(8, 1))
    model = keras.Model(inputs, layer(inputs))
    assert all(child.built for child in children)
    model.save(tmp_path / "weighted.keras")
    loaded = keras.models.load_model(tmp_path / "weighted.keras")
    choice = next(child for child in loaded.layers if isinstance(child, pp.RandomChoice))
    x = keras.ops.ones((2, 8, 1))
    for index, expected in ((0, 2.0), (1, 3.0)):
        out = choice(x, training=True, transformations={"index": keras.ops.convert_to_tensor(index, dtype="int32")})
        np.testing.assert_array_equal(array(out), np.full((2, 8, 1), expected))


@pytest.mark.parametrize("rate,count", [(0.0, 3), (1.0, 0), (0.5, 2)])
def test_random_pipeline_rate_and_inference_rng(rate, count):
    def make():
        return pp.RandomAugmentation1DPipeline(
            [pp.Rescaling1D(2.0), pp.Rescaling1D(3.0)], rate=rate, augmentations_per_sample=count, seed=42
        )

    a, b = make(), make()
    x = keras.ops.ones((2, 8, 1))
    np.testing.assert_array_equal(array(a(x, training=False)), array(x))
    actual = array(a(x, training=True))
    np.testing.assert_array_equal(actual, array(b(x, training=True)))
    if rate == 0 or count == 0:
        np.testing.assert_array_equal(actual, array(x))
    else:
        assert actual[0, 0, 0] in (1.0, 2.0, 3.0, 4.0, 6.0, 9.0)


def test_common_params_validate_before_tensor_calls(monkeypatch):
    params = pp.BaseAugmentationParams(seed=42, aligned_targets=())
    with pytest.raises(ValueError):
        pp.BaseAugmentationParams(data_format="invalid")
    layer = pp.Rescaling1D(2.0, **params.model_dump())

    def forbidden(*args, **kwargs):
        raise AssertionError("configuration validation entered tensor call")

    monkeypatch.setattr(pp.BaseAugmentationParams, "__init__", forbidden)
    np.testing.assert_array_equal(array(layer(keras.ops.ones((1, 8, 1)))), np.full((1, 8, 1), 2.0))
