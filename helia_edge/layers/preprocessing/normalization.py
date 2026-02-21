"""
# Mean/Variance Normalization Layer API

This module provides classes to build fixed mean/variance normalization layers.

Classes:
    Normalization1D: Mean/variance normalization for 1D data.
    Normalization2D: Mean/variance normalization for 2D data.
"""

import keras

from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.Normalization1D")
class Normalization1D(BaseAugmentation1D):
    mean: float | list[float] | tuple[float, ...]
    variance: float | list[float] | tuple[float, ...]
    epsilon: float

    def __init__(
        self,
        mean: float | list[float] | tuple[float, ...],
        variance: float | list[float] | tuple[float, ...],
        epsilon: float = 1e-6,
        name: str | None = None,
        **kwargs,
    ):
        """Apply fixed mean/variance normalization to 1D inputs.

        Args:
            mean: Mean value(s) used for normalization.
            variance: Variance value(s) used for normalization.
            epsilon: Small value to avoid division by zero.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Normalize a batch of samples."""
        samples = inputs[self.SAMPLES]
        stats_shape = (1, -1, 1) if self.data_format == "channels_first" else (1, 1, -1)

        mean = keras.ops.reshape(self.backend.convert_to_tensor(self.mean, dtype=samples.dtype), stats_shape)
        variance = keras.ops.reshape(
            self.backend.convert_to_tensor(self.variance, dtype=samples.dtype), stats_shape
        )
        epsilon = keras.ops.cast(self.epsilon, samples.dtype)

        return (samples - mean) / keras.ops.sqrt(variance + epsilon)

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(mean=self.mean, variance=self.variance, epsilon=self.epsilon)
        return config


@helia_export(path="helia_edge.layers.preprocessing.Normalization2D")
class Normalization2D(BaseAugmentation2D):
    mean: float | list[float] | tuple[float, ...]
    variance: float | list[float] | tuple[float, ...]
    epsilon: float

    def __init__(
        self,
        mean: float | list[float] | tuple[float, ...],
        variance: float | list[float] | tuple[float, ...],
        epsilon: float = 1e-6,
        name: str | None = None,
        **kwargs,
    ):
        """Apply fixed mean/variance normalization to 2D inputs.

        Args:
            mean: Mean value(s) used for normalization.
            variance: Variance value(s) used for normalization.
            epsilon: Small value to avoid division by zero.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Normalize a batch of samples."""
        samples = inputs[self.SAMPLES]
        stats_shape = (1, -1, 1, 1) if self.data_format == "channels_first" else (1, 1, 1, -1)

        mean = keras.ops.reshape(self.backend.convert_to_tensor(self.mean, dtype=samples.dtype), stats_shape)
        variance = keras.ops.reshape(
            self.backend.convert_to_tensor(self.variance, dtype=samples.dtype), stats_shape
        )
        epsilon = keras.ops.cast(self.epsilon, samples.dtype)

        return (samples - mean) / keras.ops.sqrt(variance + epsilon)

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(mean=self.mean, variance=self.variance, epsilon=self.epsilon)
        return config
