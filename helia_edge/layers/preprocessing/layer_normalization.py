"""
# Layer Normalization Layer API

This module provides classes to build layer normalization layers.

Classes:
    LayerNormalization1D: Layer normalization for 1D data
    LayerNormalization2D: Layer normalization for 2D data

"""

import keras
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.LayerNormalization1D")
class LayerNormalization1D(BaseAugmentation1D):
    def __init__(
        self,
        epsilon: float = 1e-6,
        name=None,
        **kwargs,
    ):
        """Apply Layer Normalization to the input.

        Args:
            epsilon (float): Small value to avoid division by zero.
        """
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment a batch of samples during training."""

        samples = inputs[self.SAMPLES]

        mean = keras.ops.mean(samples, axis=self.data_axis, keepdims=True)
        variance = keras.ops.var(samples, axis=self.data_axis, keepdims=True)
        outputs = (samples - mean) / keras.ops.sqrt(variance + self.epsilon)

        return outputs  # (batch, duration, channels)


@helia_export(path="helia_edge.layers.preprocessing.LayerNormalization2D")
class LayerNormalization2D(BaseAugmentation2D):
    def __init__(
        self,
        epsilon: float = 1e-6,
        name=None,
        **kwargs,
    ):
        """Apply Layer Normalization to the input.

        Args:
            epsilon (float): Small value to avoid division by zero.
        """
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment a batch of samples during training."""

        samples = inputs[self.SAMPLES]
        axis = (self.height_axis, self.width_axis)
        mean = keras.ops.mean(samples, axis=axis, keepdims=True)
        variance = keras.ops.var(samples, axis=axis, keepdims=True)
        outputs = (samples - mean) / keras.ops.sqrt(variance + self.epsilon)

        return outputs  # (batch, duration, channels)
