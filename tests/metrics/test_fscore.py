import numpy as np
import keras

from helia_edge.metrics import MultiF1Score


def test_multif1score_matches_keras_f1_on_2d_inputs():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)

    metric = MultiF1Score(average="macro")
    baseline = keras.metrics.F1Score(average="macro")

    metric.update_state(y_true, y_pred)
    baseline.update_state(y_true, y_pred)

    assert np.isclose(float(metric.result()), float(baseline.result()), rtol=1e-6)


def test_multif1score_flattens_higher_rank_inputs():
    y_true = np.array(
        [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.float32,
    )
    y_pred = np.array(
        [
            [[1, 0], [1, 0]],
            [[1, 0], [0, 1]],
        ],
        dtype=np.float32,
    )

    metric = MultiF1Score(average="macro")
    baseline = keras.metrics.F1Score(average="macro")

    metric.update_state(y_true, y_pred)
    baseline.update_state(y_true.reshape(-1, 2), y_pred.reshape(-1, 2))

    assert np.isclose(float(metric.result()), float(baseline.result()), rtol=1e-6)


def test_multif1score_with_sample_weight():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    y_pred = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
    sample_weight = np.array([1.0, 2.0, 1.0, 0.5], dtype=np.float32)

    metric = MultiF1Score(average="macro")
    baseline = keras.metrics.F1Score(average="macro")

    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    baseline.update_state(y_true, y_pred, sample_weight=sample_weight)

    assert np.isclose(float(metric.result()), float(baseline.result()), rtol=1e-6)
