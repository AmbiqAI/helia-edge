import numpy as np
import pytest

from helia_edge.metrics.threshold import get_predicted_threshold_indices, threshold_predictions


def test_get_predicted_threshold_indices_basic():
    y_prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]], dtype=np.float32)
    y_pred = np.array([0, 1, 0], dtype=np.int32)

    indices = get_predicted_threshold_indices(y_prob, y_pred, threshold=0.5)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2]))


def test_get_predicted_threshold_indices_strict_greater_than():
    y_prob = np.array([[0.5, 0.5], [0.49, 0.51]], dtype=np.float32)
    y_pred = np.array([0, 1], dtype=np.int32)

    indices = get_predicted_threshold_indices(y_prob, y_pred, threshold=0.5)

    np.testing.assert_array_equal(indices, np.array([1]))


def test_threshold_predictions_filters_all_arrays_consistently():
    y_prob = np.array([[0.9, 0.1], [0.45, 0.55], [0.2, 0.8]], dtype=np.float32)
    y_pred = np.array([0, 0, 1], dtype=np.int32)
    y_true = np.array([0, 1, 1], dtype=np.int32)

    out_prob, out_pred, out_true = threshold_predictions(y_prob, y_pred, y_true, threshold=0.5)

    np.testing.assert_array_equal(out_pred, np.array([0, 1]))
    np.testing.assert_array_equal(out_true, np.array([0, 1]))
    np.testing.assert_allclose(out_prob, np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_threshold_predictions_empty_result_when_none_pass():
    y_prob = np.array([[0.49, 0.51], [0.49, 0.51]], dtype=np.float32)
    y_pred = np.array([0, 0], dtype=np.int32)
    y_true = np.array([1, 1], dtype=np.int32)

    out_prob, out_pred, out_true = threshold_predictions(y_prob, y_pred, y_true, threshold=0.5)

    assert out_prob.shape == (0, 2)
    assert out_pred.shape == (0,)
    assert out_true.shape == (0,)


def test_get_predicted_threshold_indices_multiclass():
    y_prob = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    y_pred = np.array([2, 1, 0], dtype=np.int32)

    indices = get_predicted_threshold_indices(y_prob, y_pred, threshold=0.5)

    np.testing.assert_array_equal(indices, np.array([0, 2]))


def test_get_predicted_threshold_indices_shape_mismatch_raises():
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
    y_pred = np.array([0, 1, 0], dtype=np.int32)

    with pytest.raises(Exception):
        get_predicted_threshold_indices(y_prob, y_pred, threshold=0.5)
