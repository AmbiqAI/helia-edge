"""
# Confusion Matrix Plotting API

This module provides utility functions to plot confusion matrices.

Functions:
    multilabel_confusion_matrix_plot: Generate multilabel confusion matrix plot via matplotlib/seaborn
    confusion_matrix_plot: Generate confusion matrix plot via matplotlib/seaborn
    px_plot_confusion_matrix: Generate confusion matrix plot via plotly

"""

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
)


def multilabel_confusion_matrix_plot(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    save_path: os.PathLike | None = None,
    normalize: Literal["true", "pred", "all"] | None = False,
    max_cols: int = 5,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate multilabel confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.
        normalize (Literal["true", "pred", "all"] | None): Normalize. Defaults to False.
        max_cols (int): Max columns. Defaults to 5.

    Returns:
        tuple[plt.Figure, plt.Axes] | None: Figure and axes
    """
    cms = multilabel_confusion_matrix(y_true, y_pred)
    ncols = min(len(labels), max_cols)
    nrows = len(labels) // ncols + (len(labels) % ncols > 0)
    width = 10
    hdim = width / ncols
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (width, hdim * nrows)), nrows=nrows, ncols=ncols)
    for i, label in enumerate(labels):
        r = i // ncols
        c = i % ncols
        ann, fmt = True, "g"
        cm = cms[i]  # cm will be 2x2
        cmn = cm
        if normalize == "true":
            cmn = cmn / cmn.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cmn = cmn / cmn.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cmn = cmn / cmn.sum()
        cmn = np.nan_to_num(cmn)
        if normalize:
            ann = np.asarray([f"{c:g}{os.linesep}{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]).reshape(
                cm.shape
            )
            fmt = ""
        # END IF
        cm_ax = ax[c] if nrows <= 1 else ax[r][c]
        sns.heatmap(cmn, xticklabels=False, yticklabels=False, annot=ann, fmt=fmt, ax=cm_ax)
        cm_ax.set_title(label)
    # END FOR
    # Remove unused subplots
    for i in range(len(labels), nrows * ncols):
        r = i // ncols
        c = i % ncols
        cm_ax = ax[c] if nrows <= 1 else ax[r][c]
        fig.delaxes(cm_ax)
    # END FOR
    fig.text(0.5, 0.04, "Prediction", ha="center")
    fig.text(0, 0.5, "Label", va="center", rotation="vertical")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None
    # END IF
    return fig, ax


def confusion_matrix_plot(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    save_path: os.PathLike | None = None,
    normalize: Literal["true", "pred", "all"] | None = False,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes] | None:
    """Generate confusion matrix plot via matplotlib/seaborn

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        save_path (str | None): Path to save plot. Defaults to None.

    Returns:
        tuple[plt.Figure, plt.Axes] | None: Figure and axes
    """

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm
    ann = True
    fmt = "g"
    if normalize:
        cmn = confusion_matrix(y_true, y_pred, normalize=normalize)
        ann = np.asarray([f"{c:g}{os.linesep}{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]).reshape(cm.shape)
        fmt = ""
    # END IF
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))
    sns.heatmap(cmn, xticklabels=labels, yticklabels=labels, annot=ann, fmt=fmt, ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Label")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return None
    # END IF
    return fig, ax


def px_plot_confusion_matrix(
    y_true: npt.NDArray,
    y_pred: npt.NDArray,
    labels: list[str],
    normalize: Literal["true", "pred", "all"] | None = False,
    save_path: os.PathLike | None = None,
    title: str | None = None,
    width: int | None = None,
    height: int | None = 400,
    bg_color: str = "rgba(38,42,50,1.0)",
) -> go.Figure:
    """Generate confusion matrix plot via plotly

    Args:
        y_true (npt.NDArray): True y labels
        y_pred (npt.NDArray): Predicted y labels
        labels (list[str]): Label names
        normalize (Literal["true", "pred", "all"] | None): Normalize. Defaults to False.
        save_path (os.PathLike | None): Path to save plot. Defaults to None.
        title (str | None): Title. Defaults to None.
        width (int | None): Width. Defaults to None.
        height (int | None): Height. Defaults to 400.
        bg_color (str): Background color. Defaults to "rgba(38,42,50,1.0)".

    Returns:
        go.Figure: Plotly figure
    """

    cm = confusion_matrix(y_true, y_pred)
    cmn = cm
    ann = None
    if normalize:
        cmn = confusion_matrix(y_true, y_pred, normalize=normalize)
        ann = np.asarray([f"{c:g}<br>{nc:.2%}" for c, nc in zip(cm.flatten(), cmn.flatten())]).reshape(cm.shape)

    cmn = pd.DataFrame(cmn, index=labels, columns=labels)
    fig = px.imshow(
        cmn,
        labels=dict(x="Predicted", y="Actual", color="Count", text_auto=True),
        title=title,
        template="plotly_dark",
        color_continuous_scale="Plotly3",
    )
    if ann is not None:
        fig.update_traces(text=ann, texttemplate="%{text}")

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=20, r=5, t=40, b=20),
        height=height,
        width=width,
        title=title,
    )
    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn", full_html=False)

    return fig
