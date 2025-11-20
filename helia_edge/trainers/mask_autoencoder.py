"""
# Masked Autoencoder Trainer API

This module contains the implementation of a masked autoencoder trainer that
can be used to train a model using the masked autoencoder approach.

Classes:
    MaskedAutoencoder: A trainer for masked autoencoder

"""

from typing import Callable

import keras
import tensorflow as tf

from ..utils import helia_export


@helia_export(path="helia_edge.trainers.MaskedAutoencoder")
class MaskedAutoencoder(keras.Model):
    def __init__(
        self,
        patch_layer: Callable[
            [keras.KerasTensor],
            tuple[
                keras.KerasTensor,
                keras.KerasTensor,
                keras.KerasTensor,
                keras.KerasTensor,
                keras.KerasTensor,
            ],
        ],
        patch_encoder: Callable[[keras.KerasTensor], keras.KerasTensor],
        encoder: keras.Model,
        decoder: keras.Model,
        **kwargs,
    ):
        """Masked Autoencoder model for self-supervised learning.

        Args:
            patch_layer (Callable[[keras.KerasTensor], tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]]): The patch layer which will extract patches from the input.
            patch_encoder (Callable[[keras.KerasTensor], keras.KerasTensor]): The patch encoder which will encode the patches.
            encoder (keras.Model): The encoder model.
            decoder (keras.Model): The decoder model.
        """
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder

    def calculate_loss(self, x: keras.KerasTensor, test: bool = False):
        """Calculate the loss for the Masked Autoencoder model.

        Args:
            x (keras.KerasTensor): The input tensor.
            test (bool, optional): Whether the model is testing. Defaults to False.
        """
        # Patch the input.
        patches = self.patch_layer(x)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmasked patches to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = keras.ops.concatenate([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compute_loss(y=loss_patch, y_pred=loss_output)

        return total_loss, loss_patch, loss_output

    def _tensorflow_train_step(self, x):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(x)

        # Apply gradients.
        train_vars = [
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for grad, var in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        results = {}
        for metric in self.metrics:
            metric.update_state(loss_patch, loss_output)
            results[metric.name] = metric.result()
        return results

    def _tensorflow_test_step(self, x):
        total_loss, loss_patch, loss_output = self.calculate_loss(x, test=True)

        # Update the trackers.
        results = {}
        for metric in self.metrics:
            metric.update_state(loss_patch, loss_output)
            results[metric.name] = metric.result()
        return results

    def train_step(self, *args, **kwargs):
        if keras.backend.backend() == "tensorflow":
            return self._tensorflow_train_step(*args, **kwargs)
        elif keras.backend.backend() == "jax":
            raise NotImplementedError("JAX backend is not supported.")
        elif keras.backend.backend() == "torch":
            raise NotImplementedError("PyTorch backend is not supported.")

    def test_step(self, *args, **kwargs):
        if keras.backend.backend() == "tensorflow":
            return self._tensorflow_test_step(*args, **kwargs)
        elif keras.backend.backend() == "jax":
            raise NotImplementedError("JAX backend is not supported.")
        elif keras.backend.backend() == "torch":
            raise NotImplementedError("PyTorch backend is not supported.")
