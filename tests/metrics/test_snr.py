import numpy as np

from helia_edge.metrics import Snr


def _manual_snr(y_true, y_pred):
    num = np.sum(np.square(y_true))
    den = np.sum(np.square(y_pred - y_true))
    return 10.0 * np.log10(num / (den + np.finfo(np.float32).eps) + np.finfo(np.float32).eps)


def test_snr_matches_manual_single_batch_value():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.1, 2.9], dtype=np.float32)

    metric = Snr()
    metric.update_state(y_true, y_pred)

    expected = _manual_snr(y_true, y_pred)
    assert np.isclose(float(metric.result()), expected, rtol=1e-6, atol=1e-6)


def test_snr_multi_dim_input_is_flattened_correctly():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 2.2], [2.8, 4.1]], dtype=np.float32)

    metric = Snr()
    metric.update_state(y_true, y_pred)

    expected = _manual_snr(y_true.reshape(-1), y_pred.reshape(-1))
    assert np.isclose(float(metric.result()), expected, rtol=1e-6, atol=1e-6)


def test_snr_accumulates_consistently_across_batches():
    y_true_a = np.array([1.0, 2.0], dtype=np.float32)
    y_pred_a = np.array([1.0, 2.2], dtype=np.float32)
    y_true_b = np.array([3.0, 4.0], dtype=np.float32)
    y_pred_b = np.array([2.8, 4.1], dtype=np.float32)

    metric = Snr()
    metric.update_state(y_true_a, y_pred_a)
    metric.update_state(y_true_b, y_pred_b)

    y_true = np.concatenate([y_true_a, y_true_b])
    y_pred = np.concatenate([y_pred_a, y_pred_b])
    expected = _manual_snr(y_true, y_pred)
    assert np.isclose(float(metric.result()), expected, rtol=1e-6, atol=1e-6)


def test_snr_reset_state():
    metric = Snr()
    metric.update_state(np.array([1.0, 2.0], dtype=np.float32), np.array([1.0, 2.2], dtype=np.float32))

    metric.reset_state()
    metric.update_state(np.array([3.0], dtype=np.float32), np.array([3.1], dtype=np.float32))

    expected = _manual_snr(np.array([3.0], dtype=np.float32), np.array([3.1], dtype=np.float32))
    assert np.isclose(float(metric.result()), expected, rtol=1e-6, atol=1e-6)


def test_snr_get_config_round_trip():
    metric = Snr(name="snr_metric")
    restored = Snr.from_config(metric.get_config())

    assert restored.name == "snr_metric"


def test_snr_near_zero_noise_is_finite_and_large():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = y_true + 1e-6

    metric = Snr()
    metric.update_state(y_true, y_pred)
    result = float(metric.result())

    assert np.isfinite(result)
    assert result > 70.0


def test_snr_ignores_sample_weight_argument():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.1, 2.9], dtype=np.float32)
    sample_weight = np.array([10.0, 1.0, 0.5], dtype=np.float32)

    metric_no_weight = Snr()
    metric_with_weight = Snr()

    metric_no_weight.update_state(y_true, y_pred)
    metric_with_weight.update_state(y_true, y_pred, sample_weight=sample_weight)

    assert np.isclose(float(metric_no_weight.result()), float(metric_with_weight.result()), rtol=1e-6, atol=1e-6)
