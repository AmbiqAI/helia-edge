"""
# Sine Wave Layer API

This module provides classes to build sine wave layers.

Classes:
    AddSineWave: Add sine wave to the input
"""

import keras
import numpy as np
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.AddSineWave")
class AddSineWave(keras.Layer):
    sample_rate: float
    frequency: float
    amplitude: float
    data_format: str

    def __init__(
        self,
        sample_rate: float = 1,
        frequency: float = 100,
        amplitude: float = 0.1,
        data_format: str | None = None,
        **kwargs,
    ):
        """Adds a sine wave to the input.

        Args:
            sample_rate (float): Sample rate of the input.
            frequency (float): Frequency of the sine wave in Hz.
            amplitude (float): Amplitude of the sine wave.
            data_format (str): Data format of the input.
        """
        super().__init__(**kwargs)
        if sample_rate < 0:
            raise ValueError("sample_rate must be greater than 0")
        if 0 > frequency or frequency > sample_rate / 2:
            raise ValueError(" 0 <= frequency < sample_rate / 2")
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.amplitude = amplitude
        self.data_format = data_format or keras.backend.image_data_format()

        if self.data_format == "channels_first":
            self.duration_axis = -1
            self.ch_axis = -2
        else:
            self.duration_axis = -2
            self.ch_axis = -1
        # END IF

    def call(self, inputs, training=True):
        """Augment single sample with sine wave."""
        self.training = training
        is_batched = inputs.shape.rank == 3
        if not is_batched:
            inputs = keras.ops.expand_dims(inputs, axis=0)
        # END
        if self.training:
            input_shape = keras.ops.shape(inputs)
            batch_size = input_shape[0]
            duration_size = input_shape[self.duration_axis]
            ch_size = input_shape[self.ch_axis]

            # Create sine wave at the specified frequency using sample_rate as the time step
            ts = keras.ops.cast(keras.ops.arange(duration_size), self.compute_dtype) / self.sample_rate

            sine_wave = keras.ops.sin(2 * np.pi * self.frequency * ts)
            if self.data_format == "channels_first":
                sine_wave = keras.ops.reshape(sine_wave, (1, 1, duration_size))
                sine_wave = keras.ops.tile(sine_wave, (batch_size, ch_size, 1))
            else:
                sine_wave = keras.ops.reshape(sine_wave, (1, duration_size, 1))
                sine_wave = keras.ops.tile(sine_wave, (batch_size, 1, ch_size))
            outputs = keras.ops.cast(inputs + self.amplitude * sine_wave, self.compute_dtype)
        # END IF

        if not is_batched:
            outputs = keras.ops.squeeze(outputs, axis=0)
        # END IF
        return outputs

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "frequency": self.frequency,
                "amplitude": self.amplitude,
                "data_format": self.data_format,
            }
        )
        return config
