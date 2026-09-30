"""
# Rescaling Layer API

This module provides classes to build rescaling layers.

Classes:
    Rescaling1D: Rescaling 1D
    Rescaling2D: Rescaling 2D

"""

import keras

from ...utils import helia_export
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D


@helia_export(path="helia_edge.layers.preprocessing.Rescaling1D")
class Rescaling1D(BaseAugmentation1D):
    training_only = False
    scale: float

    def __init__(self, scale: float, **kwargs):
        """Rescale the input samples.

        Args:
            scale (float): The scaling factor.
        """
        super().__init__(**kwargs)
        self.scale = scale

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Rescale a batch of samples during training."""
        samples = inputs[self.SAMPLES]
        return samples * self.scale

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(scale=self.scale)
        return config


@helia_export(path="helia_edge.layers.preprocessing.Rescaling2D")
class Rescaling2D(BaseAugmentation2D):
    training_only = False
    scale: float

    def __init__(self, scale: float, **kwargs):
        """Rescale the input samples.

        Args:
            scale (float): The scaling factor.
        """
        super().__init__(**kwargs)
        self.scale = scale

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Rescale a batch of samples during training."""
        samples = inputs[self.SAMPLES]
        return samples * self.scale

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(scale=self.scale)
        return config
