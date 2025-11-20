"""
# F-Score Metrics API

This module contains additional metrics for F-Score.

Classes:
    MultiF1Score: A wrapper around keras.metrics.F1Score to handle multi-dimensional data.

"""

import keras

from ..utils import helia_export


@helia_export(path="helia_edge.metrics.MultiF1Score")
class MultiF1Score(keras.metrics.F1Score):
    """A wrapper around keras.metrics.F1Score to handle multi-dimensional data.
    This class collapses down to 2D and treats last dimension as classes.

    Example:

    ```python
    import helia_edge as helia

    f1 = helia.metrics.MultiF1Score(average='macro')
    y_true = np.array([[0, 1, 0], [1, 0, 1]])
    y_pred = np.array([[0, 1, 0], [1, 0, 0]])
    f1.update_state(y_true, y_pred)
    print(f1.result().numpy())
    ```
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = keras.ops.convert_to_tensor(y_pred, dtype=self.dtype)
        # We force data to be 2D
        if len(y_true.shape) > 2:
            y_true = keras.ops.reshape(y_true, (-1, y_true.shape[-1]))
            y_pred = keras.ops.reshape(y_pred, (-1, y_pred.shape[-1]))

        super().update_state(y_true, y_pred, sample_weight)
