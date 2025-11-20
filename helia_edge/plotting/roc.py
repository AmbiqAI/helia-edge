"""

# ROC Plotting API

This module provides utility functions to plot ROC curves.

Functions:
    roc_auc_plot: Generate ROC plot via matplotlib/seaborn

"""

import os

import matplotlib.pyplot as plt
import numpy.typing as npt
from sklearn.metrics import (
    auc,
    roc_curve,
)


def roc_auc_plot(
    y_true: npt.NDArray,
    y_prob: npt.NDArray,
    labels: list[str],
    save_path: os.PathLike | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate ROC plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_prob (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes] | None: Figure and axes
    """

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))
    label = f"ROC curve (area = {roc_auc:0.2f})"
    ax.plot(fpr, tpr, lw=2, color="darkorange", label=label)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-AUC")
    fig.legend(loc="lower right")
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return None
    return fig, ax
