"""
# Thresholding Metrics API

This module provides utility functions to threshold model predictions.

Functions:
    get_predicted_threshold_indices: Get prediction indices that are above threshold (confidence level)
    threshold_predictions: Get thresholded predictions

"""

import numpy as np
import numpy.typing as npt


def get_predicted_threshold_indices(
    y_prob: npt.NDArray,
    y_pred: npt.NDArray,
    threshold: float = 0.5,
) -> npt.NDArray:
    """Get prediction indices that are above threshold (confidence level).
    This is useful to remove weak predictions that can happen due to noisy data or poor model performance.

    Args:
        y_prob (npt.NDArray): Model output as probabilities
        y_pred (npt.NDArray, optional): Model predictions. Defaults to None.
        threshold (float): Confidence level

    Returns:
        npt.NDArray: Indices of y_prob that satisfy threshold
    """

    y_pred_prob = np.take_along_axis(y_prob, np.expand_dims(y_pred, axis=-1), axis=-1).squeeze(axis=-1)
    y_thresh_idx = np.where(y_pred_prob > threshold)[0]
    return y_thresh_idx


def threshold_predictions(
    y_prob: npt.NDArray,
    y_pred: npt.NDArray,
    y_true: npt.NDArray,
    threshold: float = 0.5,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Get prediction indices that are above threshold (confidence level).
    This is useful to remove weak predictions that can happen due to noisy data or poor model performance.

    Args:
        y_prob (npt.NDArray): Model output as probabilities
        y_pred (npt.NDArray, optional): Model predictions. Defaults to None.
        y_true (npt.NDArray): True labels
        threshold (float): Confidence level. Defaults to 0.5.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: Thresholded predictions
    """
    y_thresh_idx = get_predicted_threshold_indices(y_prob, y_pred, threshold)
    y_prob = y_prob[y_thresh_idx]
    y_pred = y_pred[y_thresh_idx]
    y_true = y_true[y_thresh_idx]
    return y_prob, y_pred, y_true
