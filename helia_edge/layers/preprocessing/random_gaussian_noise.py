"""
# Random Gaussian Noise Layer API

This module provides classes to build random Gaussian noise layers.

Classes:
    RandomGaussianNoise1D: Random Gaussian noise 1D

"""

import keras
from .base_augmentation import BaseAugmentation1D
from ...utils import parse_factor, helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomGaussianNoise1D")
class RandomGaussianNoise1D(BaseAugmentation1D):
    factor: tuple[float, float]

    def __init__(self, factor: float | tuple[float, float] = 0.1, **kwargs):
        """Apply additive zero-centered Gaussian noise.

        Args:
            factor (float): Standard deviation of the Gaussian noise.

        Example:

        ```python
            x = np.sin(2*np.pi*10*np.arange(duration_size)/100)
            lyr = RandomGaussianNoise1D(factor=0.1)
            y = lyr(x)
        ```
        """
        super().__init__(**kwargs)

        self.factor = parse_factor(factor, min_value=0, max_value=None, param_name="factor")

    def get_random_transformations(self, input_shape: tuple[int, ...]) -> dict:
        """Generate noise tensor

        Args:
            input_shape (tuple[int, ...]): Input shape.

        Returns:
            dict: Dictionary containing the noise tensor.
        """
        stddev = keras.random.uniform(
            shape=(),
            minval=self.factor[0],
            maxval=self.factor[1],
            seed=self.random_generator,
            dtype=self.compute_dtype,
        )
        return {
            "noise": keras.random.normal(
                shape=input_shape, stddev=stddev, dtype=self.compute_dtype, seed=self.random_generator
            )
        }

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment all samples in the batch as it's faster."""
        samples = inputs[self.SAMPLES]
        if self.training:
            noise = inputs[self.TRANSFORMS]["noise"]
            return samples + noise
        return samples

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
            }
        )
        return config
