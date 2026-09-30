"""Fixed sinusoidal signal addition."""

import keras
import numpy as np

from ...utils import helia_export
from .base_augmentation import BaseAugmentation1D


@helia_export(path="helia_edge.layers.preprocessing.AddSineWave")
class AddSineWave(BaseAugmentation1D):
    """Add a fixed sine wave in training and inference."""

    training_only = False

    def __init__(
        self,
        sample_rate: float = 1,
        frequency: float = 100,
        amplitude: float = 0.1,
        data_format: str | None = None,
        **kwargs,
    ):
        super().__init__(data_format=data_format, **kwargs)
        if sample_rate <= 0 or not 0 <= frequency <= sample_rate / 2:
            raise ValueError("Require positive sample_rate and 0 <= frequency <= Nyquist")
        self.sample_rate, self.frequency, self.amplitude = sample_rate, frequency, amplitude

    def augment_samples(self, inputs):
        x = inputs[self.SAMPLES]
        duration = keras.ops.shape(x)[self.data_axis]
        ts = keras.ops.arange(duration, dtype=self.compute_dtype) / self.sample_rate
        wave = self.amplitude * keras.ops.sin(2 * np.pi * self.frequency * ts)
        view = (1, 1, duration) if self.data_format == "channels_first" else (1, duration, 1)
        return x + keras.ops.reshape(wave, view)

    def get_config(self):
        return {
            **super().get_config(),
            "sample_rate": self.sample_rate,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
        }
