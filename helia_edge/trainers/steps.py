"""Training steps that dispatch on the active Keras backend."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, cast

import keras

from .._backend import Backend


class NotSupported(NotImplementedError):
    """The active Keras backend is not supported by this trainer."""


def require_backend(feature: str, supported: Sequence[str]) -> str:
    """Return the active backend, or raise ``NotSupported`` naming ``feature`` and ``supported``."""
    backend = keras.backend.backend()
    if backend not in supported:
        raise NotSupported(f"{feature} does not support the {backend!r} backend; supported: {', '.join(supported)}")
    return backend


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable gradient tracking on Torch; a no-op on other backends."""
    if keras.backend.backend() == Backend.TORCH:
        import torch

        with torch.no_grad():
            yield
    else:
        with nullcontext():
            yield


def gradient_step(
    model: keras.Model, loss_fn: Callable[[], tuple[Any, ...]], variables: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Differentiate ``loss_fn`` with the active backend and apply ``model.optimizer`` once.

    Args:
        model: Compiled model whose optimizer applies the update.
        loss_fn: Computes ``(loss, *outputs)`` from the model's current weights.
        variables: Variables to update; when None, the model's trainable weights after ``loss_fn``
            runs, so variables created by a first (building) call are included. Variables without a
            gradient are skipped.

    Returns:
        tuple: What ``loss_fn`` returned.

    Raises:
        NotSupported: On backends other than TensorFlow and Torch.
    """
    backend = require_backend("gradient_step", (Backend.TENSORFLOW, Backend.TORCH))
    if backend == Backend.TENSORFLOW:
        import tensorflow as tf

        with tf.GradientTape() as tape:
            outputs = loss_fn()
            scaled_loss = model.optimizer.scale_loss(outputs[0])
        variables = list(model.trainable_weights if variables is None else variables)
        gradients = tape.gradient(scaled_loss, variables)
        model.optimizer.apply_gradients([(g, v) for g, v in zip(gradients, variables, strict=True) if g is not None])
        return outputs

    import torch

    model.zero_grad()
    outputs = loss_fn()
    cast(torch.Tensor, model.optimizer.scale_loss(outputs[0])).backward()
    variables = list(model.trainable_weights if variables is None else variables)
    pairs = [(v.value.grad, v) for v in variables if v.value.grad is not None]
    with torch.no_grad():
        model.optimizer.apply([g for g, _ in pairs], [v for _, v in pairs])
    return outputs
