"""
# Random Noise Distortion Layer API

This module provides classes to build random noise distortion layers.

Classes:
    RandomNoiseDistortion1D: Random noise distortion 1D

"""

import keras

from .base_augmentation import BaseAugmentation1D
from ...utils import parse_factor, helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomNoiseDistortion1D")
class RandomNoiseDistortion1D(BaseAugmentation1D):
    sample_rate: float
    frequency: tuple[float, float]
    amplitude: tuple[float, float]
    interpolation: str
    noise_type: str

    def __init__(
        self,
        sample_rate: float = 1,
        frequency: float | tuple[float, float] = 100,
        amplitude: float | tuple[float, float] = 0.1,
        interpolation: str = "bilinear",
        noise_type: str = "normal",
        **kwargs,
    ):
        """Apply random noise distortion to the 1D input.
        Noise points are first generated at given frequency resolution with amplitude picked based on noise_type.
        The noise points are then interpolated to match the input duration and added to the input.

        Args:
            sample_rate (float): Sample rate of the input.
            frequency (float|tuple[float,float]): Frequency of the noise in Hz. If tuple, frequency is randomly picked between the values.
            amplitude (float|tuple[float,float]): Amplitude of the noise. If tuple, amplitude is randomly picked between the values.
            interpolation (str): Interpolation method to use. One of "nearest", "bilinear", or "bicubic".
            noise_type (str): Type of noise to generate. Currently only "normal" is supported.

        Example:

        ```python
            sample_rate = 100 # Hz
            duration = 3*sample_rate # 3 seconds
            sig_freq = 10 # Hz
            sig_amp = 1 # Signal amplitude
            noise_freq = (1, 2) # Noise frequency range
            noise_amp = (1, 2) # Noise amplitude range
            x = sig_amp*np.sin(2*np.pi*sig_freq*np.arange(duration)/sample_rate).reshape(-1, 1)
            lyr = RandomNoiseDistortion1D(sample_rate=sample_rate, frequency=noise_freq, amplitude=noise_amp)
            y = lyr(x, training=True)
        ```
        """

        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.frequency = parse_factor(frequency, min_value=None, max_value=sample_rate / 2, param_name="frequency")
        self.amplitude = parse_factor(amplitude, min_value=None, max_value=None, param_name="amplitude")
        self.interpolation = interpolation
        self.noise_type = noise_type

    def get_random_transformations(self, input_shape: tuple[int, int, int]) -> dict:
        """Generate noise distortion tensor

        Args:
            input_shape (tuple[int, ...]): Input shape.

        Returns:
            dict: Dictionary containing the noise tensor.
        """
        batch_size = input_shape[0]
        duration_size = input_shape[self.data_axis]
        ch_size = input_shape[self.ch_axis]

        # Add one period to the noise and clip later
        if self.frequency[0] == self.frequency[1]:
            frequency = self.frequency[0]
        else:
            frequency = keras.random.uniform(
                shape=(), minval=self.frequency[0], maxval=self.frequency[1], seed=self.random_generator
            )
        if self.amplitude[0] == self.amplitude[1]:
            amplitude = self.amplitude[0]
        else:
            amplitude = keras.random.uniform(
                shape=(), minval=self.amplitude[0], maxval=self.amplitude[1], seed=self.random_generator
            )

        noise_duration = keras.ops.cast((duration_size / self.sample_rate) * frequency + frequency, dtype="int32")

        if self.data_format == "channels_first":
            noise_shape = (batch_size, 1, ch_size, noise_duration)
        else:
            noise_shape = (batch_size, 1, noise_duration, ch_size)

        if self.noise_type == "normal":
            noise_pts = keras.random.normal(noise_shape, stddev=amplitude, seed=self.random_generator)
        else:
            raise ValueError(f"Invalid noise shape: {self.noise_type}")

        # keras.ops doesnt contain any low-level interpolate. So we leverage the
        # image module and fix height to 1 as workaround
        noise = keras.ops.image.resize(
            noise_pts,
            size=(1, duration_size),
            interpolation="bicubic",
            crop_to_aspect_ratio=False,
            data_format=self.data_format,
        )
        # Remove height dimension
        noise = keras.ops.squeeze(noise, axis=1)
        return {"noise": noise}

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment all samples in the batch as it's faster."""
        samples = inputs[self.SAMPLES]
        if self.training:
            noise = inputs[self.TRANSFORMS]["noise"]
            return samples + noise
        return samples

    def get_config(self):
        """Serialize the layer configuration to a JSON-compatible dictionary."""
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample,
                "frequency": self.frequency,
                "amplitude": self.amplitude,
                "noise_type": self.noise_type,
            }
        )
        return config
