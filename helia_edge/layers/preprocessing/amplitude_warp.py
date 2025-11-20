"""
# Amplitude Warp Layer

Classes:
    AmplitudeWarp: Amplitude warping layer
"""

import keras

from .base_augmentation import BaseAugmentation1D
from ...utils import parse_factor, helia_export


@helia_export(path="helia_edge.layers.preprocessing.AmplitudeWarp")
class AmplitudeWarp(BaseAugmentation1D):
    sample_rate: float
    frequency: tuple[float, float]
    amplitude: tuple[float, float]
    noise_type: str

    def __init__(
        self,
        sample_rate: float = 1,
        frequency: float | tuple[float, float] = 100,
        amplitude: float | tuple[float, float] = 0.1,
        **kwargs,
    ):
        """Apply amplitude warping to the 1D input.
        Time points are first generated at given frequency resolution with amplitude picked from uniform distribution.
        These points are then interpolated to match the input duration and multiplied to the input.

        Args:
            sample_rate (float): Sample rate of the input.
            frequency (float|tuple[float,float]): Frequency of the warping in Hz. If tuple, frequency is randomly picked between the values.
            amplitude (float|tuple[float,float]): Amplitude of the warping. If tuple, amplitude is randomly picked between the values.

        Example:

        ```python
        sample_rate = 100 # Hz
        duration = 3*sample_rate # 3 seconds
        sig_freq = 10 # Hz
        sig_amp = 1 # Signal amplitude
        noise_freq = (1, 2) # Noise frequency range
        amplitude = (0.5, 2) # Noise amplitude range
        x = sig_amp*np.sin(2*np.pi*sig_freq*np.arange(duration)/sample_rate).reshape(-1, 1).astype(np.float32)
        x = keras.ops.convert_to_tensor(x)
        import helia_edge as helia

        lyr = helia.layers.preprocessing.AmplitudeWarp(
            sample_rate=sample_rate,
            frequency=noise_freq,
            amplitude=amplitude,
        )
        y = lyr(x)
        plt.plot(x.numpy())
        plt.plot(y.numpy())
        plt.show()
        ```
        """

        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.frequency = parse_factor(frequency, min_value=None, max_value=sample_rate / 2, param_name="frequency")
        self.amplitude = parse_factor(amplitude, min_value=0, max_value=None, param_name="amplitude")

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
            frequency = keras.cast(self.frequency[0], dtype=self.compute_dtype)
        else:
            frequency = keras.random.uniform(
                shape=(),
                minval=self.frequency[0],
                maxval=self.frequency[1],
                seed=self.random_generator,
                dtype=self.compute_dtype,
            )

        duration_sec = keras.ops.cast(duration_size / self.sample_rate, dtype=self.compute_dtype)
        warp_duration = keras.ops.cast(duration_sec * frequency + frequency, dtype="int32")

        if self.data_format == "channels_first":
            warp_shape = (batch_size, 1, ch_size, warp_duration)
        else:
            warp_shape = (batch_size, 1, warp_duration, ch_size)

        warp_pts = keras.random.uniform(
            warp_shape, minval=self.amplitude[0], maxval=self.amplitude[1], seed=self.random_generator
        )

        # keras.ops doesnt contain any low-level interpolate. So we leverage the
        # image module and fix height to 1 as workaround
        warp = keras.ops.image.resize(
            warp_pts,
            size=(1, duration_size),
            interpolation="bicubic",
            crop_to_aspect_ratio=False,
            data_format=self.data_format,
        )
        # Remove height dimension
        warp = keras.ops.squeeze(warp, axis=1)
        return {"warp": warp}

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment all samples in the batch as it's faster."""
        samples = inputs[self.SAMPLES]
        if self.training:
            warp = inputs[self.TRANSFORMS]["warp"]
            return samples * warp
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
