"""
# Training History Plotting API

This module provides utility functions to plot training history metrics.

Functions:
    plot_history_metrics: Plot training history metrics

"""

from pathlib import Path
import matplotlib.pyplot as plt


def plot_history_metrics(
    history: dict[str, list[float]],
    metrics: list[str],
    save_path: Path | None = None,
    include_val: bool = True,
    figsize: tuple[int, int] = (9, 5),
    colors: tuple[str | tuple[str, str]] = ("blue", "orange"),
    stack: bool = False,
    title: str | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot training history metrics returned by model.fit.

    Args:
        history (dict[str, list[float]]): Training history
        metrics (list[str]): Metrics to plot
        save_path (Path|None, optional): Path to save plot. Defaults to None.
        include_val (bool, optional): Include validation metrics. Defaults to True.
        figsize (tuple[int, int], optional): Figure size. Defaults to (9, 5).
        colors (tuple[str | tuple[str, str]], optional): Colors for train and val. Defaults to ("blue", "orange").
        stack (bool, optional): Stack metrics. Defaults to False.
        title (str|None, optional): Title for plot. Defaults

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes handles

    Example:
    ```python

    history = dict(
        loss=[0.1, 0.2, 0.3, 0.4],
        accuracy=[0.9, 0.8, 0.7, 0.6],
        val_loss=[0.1, 0.2, 0.3, 0.4],
        val_accuracy=[0.9, 0.8, 0.7, 0.6],
    )

    import helia_edge as helia

    fig, ax = helia.plotting.plot_history_metrics(
        history,
        metrics=["loss", "accuracy"],
        include_val=True,
        stack=False,
    )

    """
    num_axes = len(metrics) if stack else 1
    fig, ax = plt.subplots(num_axes, 1, figsize=figsize, **kwargs)
    epochs = range(1, len(history["loss"]) + 1)
    for i, metric in enumerate(metrics):
        met_ax = ax[i] if stack else ax
        if len(colors) == len(metrics):
            primary_color, secondary_color = colors[i] if isinstance(colors[i], tuple) else (colors[i], colors[i])
        else:
            primary_color, secondary_color = colors if isinstance(colors, tuple) else (colors, colors)
        met_ax.plot(epochs, history[metric], color=primary_color, label="Train" if stack else metric, linestyle="--")

        if include_val:
            met_ax.plot(
                epochs,
                history[f"val_{metric}"],
                color=secondary_color,
                label="Validation" if stack else f"val_{metric}",
            )

        if stack:
            met_ax.set_ylabel(metric)
    # END FOR

    # Set x-axis label
    if num_axes > 1:
        ax[-1].set_xlabel("Epoch")
    else:
        ax.set_xlabel("Epoch")

    # Show legend if not stacked
    if not stack:
        fig.legend()

    # Set title
    if title:
        fig.suptitle(title)

    if save_path:
        fig.savefig(save_path)

    return fig, ax
