"""Masked reconstruction with portable computation and explicit TF/Torch steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import keras

from .._backend import Backend
from ..utils import helia_export
from .steps import gradient_step, no_grad, require_backend

if TYPE_CHECKING:
    from .._typing import Array, Tensor


class Reconstruction(NamedTuple):
    """Masked target patches and their corresponding predicted patches.

    Attributes:
        targets: Input patches gathered at the selected mask indices.
        predictions: Reconstructed patches gathered at the same indices.
    """

    targets: Tensor
    predictions: Tensor


class ReconstructionLoss(NamedTuple):
    """Reconstruction objective together with the patches used to compute it.

    Attributes:
        loss: Loss tensor computed by the configured reconstruction objective.
        targets: Masked input patches.
        predictions: Corresponding reconstructed patches.
    """

    loss: Tensor
    targets: Tensor
    predictions: Tensor


def _call_component[Output](component: Callable[[Array], Output], inputs: Array, training: bool) -> Output:
    # Plain callables retain their single-argument contract.
    if isinstance(component, keras.layers.Layer):
        return component(inputs, training=training)
    return component(inputs)


@helia_export(path="helia_edge.trainers.MaskedAutoencoder")
class MaskedAutoencoder(keras.Model):
    """Masked reconstruction with Keras fit() and independently callable objectives.

    call() returns (targets, predictions). training controls layer state, not masks.
    See docs/backends.md for serialization, native-loop use and support limits.
    """

    def __init__(
        self,
        patch_layer: Callable[[Array], Tensor],
        patch_encoder: Callable[[Array], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
        encoder: keras.Model,
        decoder: keras.Model,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs: Array, training: bool = False) -> Reconstruction:
        patches = _call_component(self.patch_layer, inputs, training)
        unmasked, masked, positions, mask_indices, _ = _call_component(self.patch_encoder, patches, training)
        encoded = _call_component(self.encoder, unmasked, training)
        decoder_inputs = keras.ops.concatenate([encoded + positions, masked], axis=1)
        decoded = _call_component(self.decoder, decoder_inputs, training)
        decoder_patches = _call_component(self.patch_layer, decoded, training)
        indices = keras.ops.expand_dims(mask_indices, axis=-1)
        targets = keras.ops.take_along_axis(patches, indices, axis=1)
        predictions = keras.ops.take_along_axis(decoder_patches, indices, axis=1)
        return Reconstruction(targets, predictions)

    def reconstruction_targets(self, x: Array, training: bool = False) -> Reconstruction:
        """Return masked (target_patches, predicted_patches), without an objective."""
        return self(x, training=training)

    def calculate_loss(self, x: Array, test: bool = False) -> ReconstructionLoss:
        """Return (compiled total loss, targets, predictions); preserve the old API."""
        targets, predictions = self.reconstruction_targets(x, training=not test)
        total_loss = self.compute_loss(x=x, y=targets, y_pred=predictions, training=not test)
        return ReconstructionLoss(total_loss, targets, predictions)

    def compute_loss(
        self, x: Any = None, y: Any = None, y_pred: Any = None, sample_weight: Any = None, training: bool = True
    ) -> Tensor:
        # Keras builds compiled objectives from call() before invoking train_step.
        if y is None and isinstance(y_pred, (tuple, list)):
            y, y_pred = y_pred
        return super().compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=training)

    def compute_metrics(self, x: Any, y: Any, y_pred: Any, sample_weight: Any = None) -> dict[str, Tensor]:
        if y is None and isinstance(y_pred, (tuple, list)):
            y, y_pred = y_pred
        return super().compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    @staticmethod
    def _inputs(data: Any) -> Array:
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        if y is not None:
            raise ValueError("MaskedAutoencoder generates its own targets; pass x without y.")
        if sample_weight is not None:
            raise ValueError(
                "MaskedAutoencoder does not support sample_weight; weight a native-loop objective explicitly."
            )
        return x

    def _update_metrics(
        self, loss: Tensor, targets: Tensor, predictions: Tensor, x: Array | None = None
    ) -> dict[str, Tensor]:
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss, sample_weight=keras.ops.shape(targets)[0])
                break
        # Nested layers own their metric updates.
        return self.compute_metrics(x, targets, predictions)

    def train_step(self, data: Any) -> dict[str, Tensor]:
        x = self._inputs(data)
        require_backend("MaskedAutoencoder training", (Backend.TENSORFLOW, Backend.TORCH))
        loss, targets, predictions = gradient_step(self, lambda: self.calculate_loss(x))
        with no_grad():
            return self._update_metrics(loss, targets, predictions, x=x)

    def test_step(self, data: Any) -> dict[str, Tensor]:
        x = self._inputs(data)
        require_backend("MaskedAutoencoder evaluation", (Backend.TENSORFLOW, Backend.TORCH))
        with no_grad():
            return self._update_metrics(*self.calculate_loss(x, test=True), x=x)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        for name in ("patch_layer", "patch_encoder", "encoder", "decoder"):
            config[name] = keras.saving.serialize_keras_object(getattr(self, name))
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any], custom_objects: dict[str, Any] | None = None) -> Self:
        config = dict(config)
        for name in ("patch_layer", "patch_encoder", "encoder", "decoder"):
            config[name] = keras.saving.deserialize_keras_object(config[name], custom_objects=custom_objects)
        return cls(**config)
