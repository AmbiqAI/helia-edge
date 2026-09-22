"""Plotting works without a training backend and retains public import paths."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import helia_edge as helia
from helia_edge.plotting.cm import confusion_matrix_plot
from helia_edge.plotting.history import plot_history_metrics
from helia_edge.plotting.roc import roc_auc_plot


def test_plotting_exports_and_module_paths():
    assert helia.plotting.confusion_matrix_plot is confusion_matrix_plot
    assert helia.plotting.cm.confusion_matrix_plot is confusion_matrix_plot
    assert helia.plotting.plot_history_metrics is plot_history_metrics
    assert helia.plotting.history.plot_history_metrics is plot_history_metrics
    assert helia.plotting.roc_auc_plot is roc_auc_plot
    assert helia.plotting.roc.roc_auc_plot is roc_auc_plot
    assert callable(helia.plotting.multilabel_confusion_matrix_plot)
    assert callable(helia.plotting.multilabel_confusion_matrix)
    assert callable(helia.plotting.px_plot_confusion_matrix)


def test_confusion_matrix_values_and_saved_output(tmp_path):
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    fig, ax = confusion_matrix_plot(targets, predictions, labels=["a", "b"])
    np.testing.assert_array_equal(ax.collections[0].get_array().reshape(2, 2), [[1, 1], [0, 2]])
    plt.close(fig)
    output = tmp_path / "confusion.png"
    assert confusion_matrix_plot(targets, predictions, labels=["a", "b"], save_path=output) is None
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    html = tmp_path / "confusion.html"
    plot = helia.plotting.px_plot_confusion_matrix(targets, predictions, labels=["a", "b"], save_path=html)
    np.testing.assert_array_equal(plot.data[0].z, [[1, 1], [0, 2]])
    assert "plotly" in html.read_text().lower()


def test_multilabel_counts_and_plot():
    targets = np.array([[1, 0], [0, 1], [1, 1]])
    predictions = np.array([[1, 0], [1, 1], [0, 1]])
    expected = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 2]]])
    np.testing.assert_array_equal(helia.plotting.multilabel_confusion_matrix(targets, predictions), expected)
    fig, axes = helia.plotting.multilabel_confusion_matrix_plot(targets, predictions, ["a", "b"])
    for ax, counts in zip(axes, expected):
        np.testing.assert_array_equal(ax.collections[0].get_array().reshape(2, 2), counts)
    plt.close(fig)


def test_history_and_roc_values(tmp_path):
    output = tmp_path / "history.png"
    fig, ax = plot_history_metrics({"loss": [2.0, 1.0], "val_loss": [3.0, 1.5]}, ["loss"], save_path=output)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2.0, 1.0])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [3.0, 1.5])
    assert output.read_bytes().startswith(b"\x89PNG")
    plt.close(fig)
    fig, ax = roc_auc_plot(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), ["a", "b"])
    assert "1.00" in ax.lines[0].get_label()
    plt.close(fig)
