"""Masked reconstruction with portable computation and explicit TF/Torch steps."""

from typing import Callable

import keras

from ..utils import helia_export


def _call_component(component, inputs, training):
    # Keras routes the training context even to nested layers. Plain callables
    # retain the original single-argument contract and own their state behavior.
    if isinstance(component, keras.layers.Layer):
        return component(inputs, training=training)
    return component(inputs)


@helia_export(path="helia_edge.trainers.MaskedAutoencoder")
class MaskedAutoencoder(keras.Model):
    """Train on masked patches, with an independently callable forward path.

    ``patch_layer`` maps input/reconstructed images to patches; ``patch_encoder``
    returns unmasked embeddings, masked embeddings, unmasked positions and the
    masked/unmasked indices. The decoder produces a reconstructed image in the
    same layout as the input. Plain single-argument callables remain accepted;
    use Keras layers to propagate training state and serialize their configuration.

    ``call`` and ``reconstruction_targets`` return (targets, predictions). Native
    loops may apply their own objective without compile()/fit(). Mask randomness
    is independent of the training flag. Full RNG/sampler resume, mixed precision,
    compiled Torch and distributed training are not certified by this component.
    """

    def __init__(
        self,
        patch_layer: Callable,
        patch_encoder: Callable,
        encoder: keras.Model,
        decoder: keras.Model,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        patches = _call_component(self.patch_layer, inputs, training)
        unmasked, masked, positions, mask_indices, _ = _call_component(self.patch_encoder, patches, training)
        encoded = _call_component(self.encoder, unmasked, training)
        decoder_inputs = keras.ops.concatenate([encoded + positions, masked], axis=1)
        decoded = _call_component(self.decoder, decoder_inputs, training)
        decoder_patches = _call_component(self.patch_layer, decoded, training)
        indices = keras.ops.expand_dims(mask_indices, axis=-1)
        targets = keras.ops.take_along_axis(patches, indices, axis=1)
        predictions = keras.ops.take_along_axis(decoder_patches, indices, axis=1)
        return targets, predictions

    def reconstruction_targets(self, x, training=False):
        """Return masked (target_patches, predicted_patches), without an objective."""
        return self(x, training=training)

    def calculate_loss(self, x, test=False):
        """Return (compiled total loss, targets, predictions); preserve the old API."""
        targets, predictions = self.reconstruction_targets(x, training=not test)
        total_loss = self.compute_loss(x=x, y=targets, y_pred=predictions, training=not test)
        return total_loss, targets, predictions

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        # Keras builds compiled objectives from call() before invoking train_step.
        if y is None and isinstance(y_pred, (tuple, list)):
            y, y_pred = y_pred
        return super().compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=training)

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        if y is None and isinstance(y_pred, (tuple, list)):
            y, y_pred = y_pred
        return super().compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    @staticmethod
    def _inputs(data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        if y is not None:
            raise ValueError("MaskedAutoencoder generates its own targets; pass x without y.")
        if sample_weight is not None:
            raise ValueError(
                "MaskedAutoencoder does not support sample_weight; weight a native-loop objective explicitly."
            )
        return x

    def _update_metrics(self, loss, targets, predictions, x=None):
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss, sample_weight=keras.ops.shape(targets)[0])
                break
        # Nested layers own their metric updates. Only update compiled prediction
        # metrics here, then let Keras collect all tracked metrics.
        return self.compute_metrics(x, targets, predictions)

    def _tensorflow_train_step(self, x):
        import tensorflow as tf

        with tf.GradientTape() as tape:
            loss, targets, predictions = self.calculate_loss(x)
            scaled_loss = self.optimizer.scale_loss(loss)
        variables = self.trainable_weights
        gradients = tape.gradient(scaled_loss, variables)
        pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        self.optimizer.apply_gradients(pairs)
        return self._update_metrics(loss, targets, predictions, x=x)

    def _torch_train_step(self, x):
        import torch

        self.zero_grad()
        loss, targets, predictions = self.calculate_loss(x)
        self.optimizer.scale_loss(loss).backward()
        variables = self.trainable_weights
        pairs = [(v.value.grad, v) for v in variables if v.value.grad is not None]
        with torch.no_grad():
            self.optimizer.apply([g for g, _ in pairs], [v for _, v in pairs])
            return self._update_metrics(loss, targets, predictions, x=x)

    def train_step(self, data):
        x = self._inputs(data)
        backend = keras.backend.backend()
        if backend == "tensorflow":
            return self._tensorflow_train_step(x)
        if backend == "torch":
            return self._torch_train_step(x)
        raise NotImplementedError(f"MaskedAutoencoder training does not support the {backend!r} backend.")

    def test_step(self, data):
        x = self._inputs(data)
        backend = keras.backend.backend()
        if backend == "torch":
            import torch

            with torch.no_grad():
                return self._update_metrics(*self.calculate_loss(x, test=True), x=x)
        if backend == "tensorflow":
            return self._update_metrics(*self.calculate_loss(x, test=True), x=x)
        raise NotImplementedError(f"MaskedAutoencoder evaluation does not support the {backend!r} backend.")

    def get_config(self):
        config = super().get_config()
        for name in ("patch_layer", "patch_encoder", "encoder", "decoder"):
            config[name] = keras.saving.serialize_keras_object(getattr(self, name))
        return config

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        for name in ("patch_layer", "patch_encoder", "encoder", "decoder"):
            config[name] = keras.saving.deserialize_keras_object(config[name])
        return cls(**config)
