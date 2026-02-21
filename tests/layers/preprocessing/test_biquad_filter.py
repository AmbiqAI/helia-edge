import numpy as np
import scipy.signal
import tensorflow as tf

from helia_edge.layers.preprocessing import CascadedBiquadFilter
from helia_edge.layers.preprocessing.biquad_filter import get_butter_sos


def test_cascaded_biquad_filter_matches_scipy_sosfilt_channels_last():
    t = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    x = np.stack(
        [
            np.sin(2 * np.pi * 10 * t) + 0.25 * np.sin(2 * np.pi * 80 * t),
            np.sin(2 * np.pi * 15 * t) + 0.15 * np.sin(2 * np.pi * 90 * t),
        ],
        axis=-1,
    ).astype(np.float32)

    layer = CascadedBiquadFilter(lowcut=5.0, highcut=25.0, sample_rate=256.0, order=3, data_format="channels_last")
    y = np.asarray(layer(tf.constant(x), training=True))

    sos = get_butter_sos(lowcut=5.0, highcut=25.0, sample_rate=256.0, order=3)
    y_ref = scipy.signal.sosfilt(sos, x, axis=0).astype(np.float32)
    np.testing.assert_allclose(y, y_ref, rtol=1e-5, atol=1e-5)


def test_cascaded_biquad_filter_channels_first_matches_channels_last():
    t = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    x_last = np.stack(
        [
            np.sin(2 * np.pi * 10 * t) + 0.25 * np.sin(2 * np.pi * 80 * t),
            np.sin(2 * np.pi * 15 * t) + 0.15 * np.sin(2 * np.pi * 90 * t),
        ],
        axis=-1,
    ).astype(np.float32)
    x_first = np.transpose(x_last, (1, 0))

    layer_last = CascadedBiquadFilter(lowcut=5.0, highcut=25.0, sample_rate=256.0, order=3, data_format="channels_last")
    layer_first = CascadedBiquadFilter(
        lowcut=5.0, highcut=25.0, sample_rate=256.0, order=3, data_format="channels_first"
    )
    y_last = np.asarray(layer_last(tf.constant(x_last), training=True))
    y_first = np.asarray(layer_first(tf.constant(x_first), training=True))
    y_first_as_last = np.transpose(y_first, (1, 0))

    np.testing.assert_allclose(y_first_as_last, y_last, rtol=1e-6, atol=1e-6)


def test_cascaded_biquad_filter_compiles_with_tf_function():
    t = np.linspace(0.0, 1.0, 256, endpoint=False, dtype=np.float32)
    x = np.stack(
        [
            np.sin(2 * np.pi * 10 * t) + 0.25 * np.sin(2 * np.pi * 80 * t),
            np.sin(2 * np.pi * 15 * t) + 0.15 * np.sin(2 * np.pi * 90 * t),
        ],
        axis=-1,
    ).astype(np.float32)

    layer = CascadedBiquadFilter(
        lowcut=5.0,
        highcut=25.0,
        sample_rate=256.0,
        order=3,
        forward_backward=True,
        data_format="channels_last",
    )
    @tf.function
    def run_filter(z):
        return layer(z, training=True)

    y_fb = np.asarray(run_filter(tf.constant(x)))
    assert y_fb.shape == x.shape
    assert np.isfinite(y_fb).all()


def test_cascaded_biquad_filter_get_config_round_trip():
    layer = CascadedBiquadFilter(
        lowcut=8.0,
        highcut=20.0,
        sample_rate=200.0,
        order=4,
        forward_backward=True,
        name="biquad_test",
    )

    restored = CascadedBiquadFilter.from_config(layer.get_config())

    assert restored.lowcut == 8.0
    assert restored.highcut == 20.0
    assert restored.sample_rate == 200.0
    assert restored.order == 4
    assert restored.forward_backward is True
    assert restored.name == "biquad_test"


def test_get_butter_sos_cache_returns_same_coefficients():
    sos_a = get_butter_sos(lowcut=5.0, highcut=25.0, sample_rate=256.0, order=3)
    sos_b = get_butter_sos(lowcut=5.0, highcut=25.0, sample_rate=256.0, order=3)
    np.testing.assert_array_equal(sos_a, sos_b)
