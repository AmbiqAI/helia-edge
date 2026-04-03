import math

import keras
import numpy as np

from helia_edge.metrics import PRD, TruePRD


def test_prd_normalized_matches_expected_value():
    metric = PRD(normalized=True)
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 1.0, 5.0], dtype=np.float32)

    metric.update_state(y_true, y_pred)

    expected = 100.0 * math.sqrt(5.0 / 14.0)
    assert np.isclose(float(metric.result()), expected, rtol=1e-6)


def test_prd_unnormalized_matches_expected_value():
    metric = PRD(normalized=False)
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 1.0, 5.0], dtype=np.float32)

    metric.update_state(y_true, y_pred)

    expected = 100.0 * math.sqrt(5.0 / 3.0)
    assert np.isclose(float(metric.result()), expected, rtol=1e-6)


def test_prd_supports_sample_weight():
    metric = PRD(normalized=True)
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 1.0, 5.0], dtype=np.float32)
    sample_weight = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    expected = 100.0 * math.sqrt(4.0 / 10.0)
    assert np.isclose(float(metric.result()), expected, rtol=1e-6)


def test_true_prd_is_normalized_prd():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 1.0, 5.0], dtype=np.float32)

    prd = PRD(normalized=True)
    true_prd = TruePRD()

    prd.update_state(y_true, y_pred)
    true_prd.update_state(y_true, y_pred)

    assert np.isclose(float(true_prd.result()), float(prd.result()), rtol=1e-6)


def test_prd_reset_state_clears_accumulators():
    metric = PRD(normalized=True)
    metric.update_state(np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))
    assert float(metric.result()) > 0.0

    metric.reset_state()
    assert np.isclose(float(metric.result()), 0.0, atol=1e-7)


def test_prd_get_config_round_trip_and_registration_package():
    metric = PRD(normalized=False, name="my_prd")
    config = metric.get_config()
    restored = PRD.from_config(config)

    assert restored.normalized is False
    assert restored.name == "my_prd"

    serialized = keras.saving.serialize_keras_object(metric)
    assert serialized["registered_name"] == "metrics>PRD"


def test_prd_accumulates_across_multiple_update_state_calls():
    metric = PRD(normalized=True)

    y_true_a = np.array([1.0, 2.0], dtype=np.float32)
    y_pred_a = np.array([1.0, 2.2], dtype=np.float32)
    y_true_b = np.array([3.0, 4.0], dtype=np.float32)
    y_pred_b = np.array([2.8, 4.1], dtype=np.float32)

    metric.update_state(y_true_a, y_pred_a)
    metric.update_state(y_true_b, y_pred_b)

    y_true = np.concatenate([y_true_a, y_true_b])
    y_pred = np.concatenate([y_pred_a, y_pred_b])
    expected = 100.0 * math.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum(y_true**2))

    assert np.isclose(float(metric.result()), expected, rtol=1e-6, atol=1e-6)


def test_prd_zero_error_returns_zero():
    metric = PRD(normalized=True)
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    metric.update_state(y_true, y_true)

    assert np.isclose(float(metric.result()), 0.0, atol=1e-7)


def test_prd_handles_zero_denominator_normalized():
    metric = PRD(normalized=True)
    y_true = np.zeros((3,), dtype=np.float32)
    y_pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    metric.update_state(y_true, y_pred)

    result = float(metric.result())
    assert np.isfinite(result)


def test_true_prd_registration_name():
    serialized = keras.saving.serialize_keras_object(TruePRD())
    assert serialized["registered_name"] == "metrics>TruePRD"


def test_prd_unnormalized_with_sample_weight_uses_weight_sum_denominator():
    metric = PRD(normalized=False)
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 1.0, 5.0], dtype=np.float32)
    sample_weight = np.array([1.0, 2.0, 1.0], dtype=np.float32)

    metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    expected = 100.0 * math.sqrt(6.0 / 4.0)
    assert np.isclose(float(metric.result()), expected, rtol=1e-6, atol=1e-6)


def test_true_prd_config_round_trip():
    metric = TruePRD(name="my_true_prd")
    restored = TruePRD.from_config(metric.get_config())

    assert restored.name == "my_true_prd"
    assert restored.normalized is True
