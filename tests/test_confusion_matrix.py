import numpy as np
import keras

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
