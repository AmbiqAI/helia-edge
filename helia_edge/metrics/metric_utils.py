"""
# Metrics Utils API

This module provides utility functions to compute metrics for classification tasks.

Functions:
    compute_metrics: Compute set of metrics for y_true and y_pred
    confusion_matrix: Compute confusion matrix using keras w/ addition to normalize

"""

import logging
from typing import Literal

import keras
from keras.src.metrics import metrics_utils

logger = logging.getLogger(__name__)


def compute_metrics(
    metrics: list[keras.Metric], y_true: keras.KerasTensor, y_pred: keras.KerasTensor
) -> dict[str, float]:
    """Compute set of metrics for y_true and y_pred.

    Args:
        metrics (list[keras.Metric]): List of metrics
        y_true (keras.KerasTensor): True labels
        y_pred (keras.KerasTensor): Predicted labels

    Returns:
        dict: Dictionary of metric names and values

    Example:

    ```python
    import helia_edge as helia

    metrics = [keras.metrics.Accuracy('acc'), keras.metrics.Precision(0.5, name='precision')]
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    results = helia.metrics.compute_metrics(metrics, y_true, y_pred)
    print(results)
    ```
    """

    results = {}
    for metric in metrics:
        metric.reset_state()
        metric.update_state(y_true, y_pred)
        results[metric.name] = metric.result().numpy()
    return results


def confusion_matrix(
    labels: keras.KerasTensor,
    predictions: keras.KerasTensor,
    num_classes: int,
    weights: keras.KerasTensor | None = None,
    dtype: str = "int32",
    normalize: Literal["true", "pred", "all"] | None = None,
) -> keras.KerasTensor:
    """Compute confusion matrix using keras w/ addition to normalize.

    Normalization modes:
        - "true": Normalize by true labels
        - "pred": Normalize by predicted labels
        - "all": Normalize by all labels

    Args:
        labels (keras.KerasTensor): True labels
        predictions (keras.KerasTensor): Predicted labels
        num_classes (int): Number of classes
        weights (keras.KerasTensor, optional): Weights. Defaults to None.
        dtype (str, optional): Data type. Defaults to "int32".
        normalize (Literal["true", "pred", "all"], optional): Normalization mode. Defaults to None.

    Returns:
        keras.KerasTensor: Confusion matrix

    Example:

    ```python
    import helia_edge as helia

    labels = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 1])
    num_classes = 2
    cm = helia.metrics.confusion_matrix(labels, predictions, num_classes, normalize='true')
    print(cm)
    ```
    """
    cm = metrics_utils.confusion_matrix(
        labels,
        predictions,
        num_classes,
        weights=weights,
        dtype=dtype,
    )

    if normalize == "true":
        cm = keras.ops.divide(cm, keras.ops.sum(cm, axis=1, keepdims=True))
    elif normalize == "pred":
        cm = keras.ops.divide(cm, keras.ops.sum(cm, axis=0, keepdims=True))
    elif normalize == "all":
        cm = keras.ops.divide(cm, keras.ops.sum(cm))
    cm = keras.ops.nan_to_num(cm, posinf=0, neginf=0)
    return cm
