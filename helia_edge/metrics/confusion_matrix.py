"""
# Confusion Matrix Metric API

Classes:
    ConfusionMatrix: Accumulates a confusion matrix and returns row-normalized values.
"""

import keras
from keras.src.metrics import metrics_utils

from ..utils import helia_export


@helia_export(path="helia_edge.metrics.ConfusionMatrix")
class ConfusionMatrix(keras.metrics.Metric):
    def __init__(self, num_classes: int, name="confusion_matrix", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.conf_matrix = self.add_variable(
            name="conf_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype="int64",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_pred: shape (batch, ..., num_classes) or integer class labels
        y_true: shape (batch, ...) with integer class labels
        """
        y_true = keras.ops.convert_to_tensor(y_true)
        y_pred = keras.ops.convert_to_tensor(y_pred)

        if y_pred.shape is not None and len(y_pred.shape) > 1 and y_pred.shape[-1] == self.num_classes:
            pred_labels = keras.ops.argmax(y_pred, axis=-1)
        else:
            pred_labels = y_pred

        y_true_flat = keras.ops.reshape(y_true, (-1,))
        pred_flat = keras.ops.reshape(pred_labels, (-1,))

        if sample_weight is not None:
            sample_weight = keras.ops.convert_to_tensor(sample_weight)
            sample_weight = keras.ops.reshape(sample_weight, (-1,))

        batch_conf_matrix = metrics_utils.confusion_matrix(
            labels=y_true_flat,
            predictions=pred_flat,
            num_classes=self.num_classes,
            weights=sample_weight,
            dtype="int64",
        )

        self.conf_matrix.assign_add(batch_conf_matrix)

    def result(self):
        """Returns the row-normalized confusion matrix as float64."""
        conf_matrix_f64 = keras.ops.cast(self.conf_matrix, "float64")
        row_sums = keras.ops.sum(conf_matrix_f64, axis=1, keepdims=True)
        normalized = keras.ops.divide(conf_matrix_f64, row_sums)
        return keras.ops.nan_to_num(normalized, posinf=0, neginf=0)

    def reset_state(self):
        for v in self.variables:
            v.assign(keras.ops.zeros(v.shape, dtype=v.dtype))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config
