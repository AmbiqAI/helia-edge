"""
# :material-chart-bar: Plotting Module API

This module provides utility functions to plot metrics and confusion matrices.

Functions:
    confusion_matrix_plot: Plot confusion matrix
    px_plot_confusion_matrix: Plot confusion matrix using Plotly
    multilabel_confusion_matrix: Compute multilabel confusion matrix
    multilabel_confusion_matrix_plot: Plot multilabel confusion matrix
    roc_auc_plot: Plot ROC-AUC curve
    plot_history_metrics: Plot training history metrics

"""

from .cm import (
    confusion_matrix_plot,
    px_plot_confusion_matrix,
    multilabel_confusion_matrix,
    multilabel_confusion_matrix_plot,
)
from .roc import roc_auc_plot
from .history import plot_history_metrics
