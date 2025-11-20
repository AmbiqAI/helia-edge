"""
# Resizing layers API

This module provides classes to build resizing layers.

Classes:
    Resizing1D: Resize 1D samples
    Resizing2D: Resize 2D samples
"""

import keras
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.Resizing1D")
class Resizing1D(BaseAugmentation1D):
    duration: int
    data_format: str

    def __init__(self, duration: int, **kwargs):
        """1D resizing layer

        Args:
            duration (int): The new duration of the samples

        """
        super().__init__(**kwargs)
        self.duration = duration

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Resize a batch of samples during training."""
        samples = inputs[self.SAMPLES]

        # Add height dimension
        samples = keras.ops.expand_dims(samples, axis=1)
        samples = keras.ops.image.resize(
            samples,
            size=(1, self.duration),
            interpolation="bicubic",
            crop_to_aspect_ratio=False,
            data_format=self.data_format,
        )
        # Remove height dimension
        samples = keras.ops.squeeze(samples, axis=1)
        return samples

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[self.data_axis] = self.duration
        return tuple(output_shape)

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(
            duration=self.duration,
            data_format=self.data_format,
        )
        return config


@helia_export(path="helia_edge.layers.preprocessing.Resizing2D")
class Resizing2D(BaseAugmentation2D):
    height: int
    width: int
    interpolation: str

    def __init__(self, height: int, width: int, interpolation: str = "bicubic", **kwargs):
        """"""
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Resize a batch of samples during training."""
        samples = inputs[self.SAMPLES]
        samples = keras.ops.image.resize(
            samples,
            size=(self.height, self.width),
            interpolation="bicubic",
            crop_to_aspect_ratio=False,
            data_format=self.data_format,
        )
        return samples

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[self.height_axis] = self.height
        output_shape[self.width_axis] = self.width
        return tuple(output_shape)

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(
            height=self.height,
            width=self.width,
        )
        return config
