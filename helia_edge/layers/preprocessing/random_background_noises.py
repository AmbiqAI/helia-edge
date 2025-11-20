"""
# Random Background Noises Layer API

This module provides classes to build random background noises layers.

Classes:
    RandomBackgroundNoises1D: Random background noises 1D


"""

import keras

from .base_augmentation import BaseAugmentation1D
from ...utils import parse_factor, helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomBackgroundNoises1D")
class RandomBackgroundNoises1D(BaseAugmentation1D):
    amplitude: tuple[float, float]
    num_noises: int

    def __init__(self, noises, amplitude: float | tuple[float, float] = 0.1, num_noises: int = 1, **kwargs):
        """Apply random background noises to the input.

        Args:
            noises (np.ndarray): Background noises to apply.
            amplitude (float|tuple[float,float]): Amplitude of the noise. If tuple, amplitude is randomly picked between the values.

        Example:

        ```python
            sample_rate = 100
            duration = 2*sample_rate
            freqs = [2, 5, 15, 20, 25]
            noises = np.vstack([
                np.sin(2*np.pi*f*np.arange(duration)/sample_rate)
                for f in freqs
            ]).T
            lyr = RandomBackgroundNoises(noises=noises, amplitude=0.2, num_noises=2)
            y = lyr(x, training=True)
        ```
        """
        super().__init__(**kwargs)

        self.amplitude = parse_factor(amplitude, min_value=0, max_value=None, param_name="amplitude")
        self.num_noises = num_noises
        self.noises = noises

    def get_random_transformations(self, input_shape: tuple[int, int, int]) -> dict:
        """Generate noise tensor

        Args:
            input_shape (tuple[int, ...]): Input shape.

        Returns:
            dict: Dictionary containing the noise tensor.
        """
        batch_size = input_shape[0]
        duration_size = input_shape[self.data_axis]

        noise_idx = keras.random.randint(
            shape=(batch_size, self.num_noises),
            minval=0,
            maxval=self.noises.shape[1],
        )
        start = keras.random.randint(
            shape=(batch_size, self.num_noises),
            minval=0,
            maxval=self.noises.shape[0] - duration_size + 1,
            seed=self.random_generator,
            dtype="int32",
        )
        amplitude = keras.random.uniform(
            shape=(batch_size, self.num_noises),
            minval=self.amplitude[0],
            maxval=self.amplitude[1],
            seed=self.random_generator,
            dtype=self.compute_dtype,
        ) / keras.ops.cast(self.num_noises, self.compute_dtype)

        return {
            "noise_idx": noise_idx,
            "start": start,
            "amplitude": amplitude,
        }

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Augment single sample with random background noises."""
        duration_size = inputs[self.SAMPLES].shape[self.data_axis]
        ch_size = inputs[self.SAMPLES].shape[self.ch_axis]

        def random_noise(i, x):
            noise_idx = inputs[self.TRANSFORMS]["noise_idx"][i]
            start = inputs[self.TRANSFORMS]["start"][i]
            amplitude = inputs[self.TRANSFORMS]["amplitude"][i]
            noise = keras.ops.slice(self.noises, (start, noise_idx), (duration_size, 1))
            noise = keras.ops.squeeze(noise)
            if self.data_format == "channels_first":
                noise = keras.ops.reshape(noise, (1, duration_size))
                noise = keras.ops.tile(noise, (ch_size, 1))
            else:
                noise = keras.ops.reshape(noise, (duration_size, 1))
                noise = keras.ops.tile(noise, (1, ch_size))
            return x + amplitude * noise

        # END DEF

        if self.training:
            sample = inputs[self.SAMPLES]
            outputs = keras.ops.fori_loop(lower=0, upper=self.num_noises, body_fun=random_noise, init_val=sample)
        else:
            outputs = inputs[self.SAMPLES]
        return outputs

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "amplitude": self.amplitude,
                "num_noises": self.num_noises,
            }
        )
        return config
