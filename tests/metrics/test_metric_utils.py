import numpy as np
import keras

from helia_edge.metrics.metric_utils import compute_metrics, confusion_matrix


def test_compute_metrics_resets_each_metric_before_use():
    metric = keras.metrics.Accuracy(name="acc")

    metric.update_state(np.array([0, 1]), np.array([1, 1]))

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    results = compute_metrics([metric], y_true, y_pred)

    assert np.isclose(results["acc"], 0.75, rtol=1e-6)


def test_compute_metrics_returns_named_results():
    metrics = [keras.metrics.Accuracy(name="acc"), keras.metrics.Precision(name="prec")]
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    results = compute_metrics(metrics, y_true, y_pred)

    assert set(results.keys()) == {"acc", "prec"}


def test_compute_metrics_supports_tensor_valued_metric_results():
    metrics = [keras.metrics.Precision(name="prec_vec", thresholds=[0.3, 0.7])]
    y_true = np.array([0, 1, 1, 0], dtype=np.float32)
    y_pred = np.array([0.1, 0.9, 0.4, 0.8], dtype=np.float32)

    results = compute_metrics(metrics, y_true, y_pred)

    assert isinstance(results["prec_vec"], np.ndarray)
    assert results["prec_vec"].shape == (2,)


def test_compute_metrics_supports_multiple_same_metric_class_names():
    metrics = [keras.metrics.Accuracy(name="acc_a"), keras.metrics.Accuracy(name="acc_b")]
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    results = compute_metrics(metrics, y_true, y_pred)

    assert set(results.keys()) == {"acc_a", "acc_b"}
    assert np.isclose(results["acc_a"], results["acc_b"], rtol=1e-6)


def test_confusion_matrix_no_normalize_counts():
    labels = np.array([0, 1, 1, 0], dtype=np.int32)
    predictions = np.array([0, 1, 0, 1], dtype=np.int32)

    cm = confusion_matrix(labels, predictions, num_classes=2, dtype="float32")

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(cm),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_confusion_matrix_normalize_true_rows_sum_to_one_or_zero():
    labels = np.array([0, 1, 1, 0], dtype=np.int32)
    predictions = np.array([0, 1, 0, 1], dtype=np.int32)

    cm = confusion_matrix(labels, predictions, num_classes=3, dtype="float32", normalize="true")
    cm_np = keras.ops.convert_to_numpy(cm)
    row_sums = cm_np.sum(axis=1)

    np.testing.assert_allclose(row_sums[:2], np.ones(2), rtol=1e-6, atol=1e-6)
    assert np.isclose(row_sums[2], 0.0, atol=1e-6)


def test_confusion_matrix_normalize_pred_columns_sum_to_one_or_zero():
    labels = np.array([0, 1, 1, 0], dtype=np.int32)
    predictions = np.array([0, 1, 0, 1], dtype=np.int32)

    cm = confusion_matrix(labels, predictions, num_classes=3, dtype="float32", normalize="pred")
    cm_np = keras.ops.convert_to_numpy(cm)
    col_sums = cm_np.sum(axis=0)

    np.testing.assert_allclose(col_sums[:2], np.ones(2), rtol=1e-6, atol=1e-6)
    assert np.isclose(col_sums[2], 0.0, atol=1e-6)


def test_confusion_matrix_normalize_all_sums_to_one():
    labels = np.array([0, 1, 1, 0], dtype=np.int32)
    predictions = np.array([0, 1, 0, 1], dtype=np.int32)

    cm = confusion_matrix(labels, predictions, num_classes=2, dtype="float32", normalize="all")

    assert np.isclose(float(keras.ops.convert_to_numpy(keras.ops.sum(cm))), 1.0, atol=1e-6)


def test_confusion_matrix_weighted_float_dtype_support():
    labels = np.array([0, 0], dtype=np.int32)
    predictions = np.array([0, 1], dtype=np.int32)
    weights = np.array([0.25, 0.75], dtype=np.float32)

    cm = confusion_matrix(labels, predictions, num_classes=2, weights=weights, dtype="float32")
    cm_np = keras.ops.convert_to_numpy(cm)

    np.testing.assert_allclose(cm_np, np.array([[0.25, 0.75], [0.0, 0.0]], dtype=np.float32), rtol=1e-6, atol=1e-6)
    assert np.isfinite(cm_np).all()
