"""
# Random Sine Wave Layer API

This module provides classes to build random sine wave layers.

Classes:
    RandomSineWave: Random sine wave

"""

import keras
import numpy as np

from .base_augmentation import BaseAugmentation1D
from ...utils import parse_factor, helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomSineWave")
class RandomSineWave(BaseAugmentation1D):
    sample_rate: float
    frequency: tuple[float, float]
    amplitude: tuple[float, float]
    data_format: str

    def __init__(
        self,
        sample_rate: float = 1,
        frequency: float | tuple[float, float] = 100,
        amplitude: float | tuple[float, float] = 0.1,
        **kwargs,
    ):
        """Adds a sine wave to the input.

        Args:
            sample_rate (float): Sample rate of the input.
            frequency (float|tuple[float,float]): Frequency of the wave in Hz. If tuple, frequency is randomly picked between the values.
            amplitude (float|tuple[float,float]): Amplitude of the wave. If tuple, amplitude is randomly picked between the values.
        """
        super().__init__(**kwargs)
        if sample_rate < 0:
            raise ValueError("sample_rate must be greater than 0")
        self.sample_rate = sample_rate
        self.frequency = parse_factor(frequency, min_value=0, max_value=sample_rate / 2, param_name="frequency")
        self.amplitude = parse_factor(amplitude, min_value=0, max_value=None, param_name="amplitude")

    def get_random_transformations(self, input_shape: tuple[int, int, int]) -> dict:
        """Generate noise distortion tensor

        Args:
            input_shape (tuple[int, ...]): Input shape.

        Returns:
            dict: Dictionary containing the noise tensor.
        """
        batch_size = input_shape[0]

        frequencies = keras.random.uniform(
            shape=(batch_size,), minval=self.frequency[0], maxval=self.frequency[1], seed=self.random_generator
        )
        amplitudes = keras.random.uniform(
            shape=(batch_size,), minval=self.amplitude[0], maxval=self.amplitude[1], seed=self.random_generator
        )

        return {"frequency": frequencies, "amplitude": amplitudes}

    def augment_sample(self, inputs):
        """Augment single sample with sine wave."""
        sample = inputs[self.SAMPLES]
        duration_size = sample.shape[self.data_axis]
        sample_rate = keras.ops.cast(self.sample_rate, dtype=self.compute_dtype)
        if self.training:
            frequency = inputs[self.TRANSFORMS]["frequency"]
            amplitude = inputs[self.TRANSFORMS]["amplitude"]
            ts = keras.ops.arange(duration_size, dtype=self.compute_dtype) / sample_rate
            sine_wave = keras.ops.sin(2 * np.pi * frequency * ts)
            sine_wave = amplitude * sine_wave
            sine_wave = keras.ops.reshape(sine_wave, sample.shape)
            return sample + sine_wave
        return sample

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "frequency": self.frequency,
                "amplitude": self.amplitude,
            }
        )
        return config
