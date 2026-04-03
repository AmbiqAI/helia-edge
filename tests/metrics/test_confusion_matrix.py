import numpy as np
import keras
import pytest

import helia_edge as helia


def test_confusion_matrix_probs_normalized():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )

    metric.update_state(y_true, y_pred)
    result = keras.ops.convert_to_numpy(metric.result())

    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_confusion_matrix_sample_weight():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)
    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])
    sample_weight = np.array([2, 1])

    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    result = keras.ops.convert_to_numpy(metric.result())

    expected = np.array([[2.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_confusion_matrix_fractional_sample_weight():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)
    y_true = np.array([0, 0], dtype=np.int32)
    y_pred = np.array([0, 1], dtype=np.int32)
    sample_weight = np.array([0.25, 0.75], dtype=np.float32)

    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    result = keras.ops.convert_to_numpy(metric.result())

    expected = np.array([[0.25, 0.75], [0.0, 0.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_confusion_matrix_accumulates_multiple_batches_then_normalizes():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)

    metric.update_state(np.array([0, 1]), np.array([0, 1]))
    metric.update_state(np.array([0, 1]), np.array([1, 0]))

    result = keras.ops.convert_to_numpy(metric.result())

    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_confusion_matrix_one_hot_y_pred_path():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)
    y_true = np.array([0, 1, 1, 0], dtype=np.int32)
    y_pred = np.array(
        [
            [0.7, 0.3],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )

    metric.update_state(y_true, y_pred)
    result = keras.ops.convert_to_numpy(metric.result())

    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_confusion_matrix_reset_state_returns_zero_matrix():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)
    metric.update_state(np.array([0, 1]), np.array([0, 1]))

    metric.reset_state()
    result = keras.ops.convert_to_numpy(metric.result())

    np.testing.assert_allclose(result, np.zeros((2, 2)), rtol=1e-6, atol=1e-6)


def test_confusion_matrix_config_round_trip():
    metric = helia.metrics.ConfusionMatrix(num_classes=3, name="cm")

    restored = helia.metrics.ConfusionMatrix.from_config(metric.get_config())

    assert restored.num_classes == 3
    assert restored.name == "cm"


def test_confusion_matrix_handles_empty_row_without_nan():
    metric = helia.metrics.ConfusionMatrix(num_classes=3)

    metric.update_state(np.array([0, 0, 1]), np.array([0, 1, 1]))
    result = keras.ops.convert_to_numpy(metric.result())

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[2], np.zeros(3), rtol=1e-6, atol=1e-6)


def test_confusion_matrix_honors_metric_dtype():
    metric = helia.metrics.ConfusionMatrix(num_classes=2, dtype="float64")
    metric.update_state(np.array([0, 1]), np.array([0, 1]))

    assert "float64" in str(metric.conf_matrix.dtype)


def test_confusion_matrix_invalid_label_raises():
    metric = helia.metrics.ConfusionMatrix(num_classes=2)

    with pytest.raises(ValueError, match=r"labels and predictions must be in \[0, 1\]"):
        metric.update_state(np.array([0, 2]), np.array([0, 1]))
