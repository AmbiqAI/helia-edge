"""Plotting helpers, loaded on first use with the ``plotting`` extra.

Functions:
    confusion_matrix_plot: Plot a confusion matrix.
    px_plot_confusion_matrix: Plot an interactive confusion matrix.
    multilabel_confusion_matrix: Compute per-label confusion matrices.
    multilabel_confusion_matrix_plot: Plot per-label confusion matrices.
    roc_auc_plot: Plot a ROC curve.
    plot_history_metrics: Plot training history metrics.
"""

from helia_edge._lazy import attach_exports as _attach_exports

__getattr__, __dir__, __all__ = _attach_exports(__name__, __file__)
