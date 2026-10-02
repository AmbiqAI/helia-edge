"""
# Spectral Layers API

Classes:
    StftMagnitude: Magnitude of a short-time Fourier transform with a fixed, stored basis
"""

import keras

from ..utils.export import helia_export


@helia_export(path="helia_edge.layers.StftMagnitude")
class StftMagnitude(keras.layers.Layer):
    """Magnitude of a short-time Fourier transform computed as a strided convolution with a stored basis.

    The input ``(batch, samples)`` is reflect-padded by ``padding`` (start, end) samples and convolved
    with ``basis`` (``frame_length`` taps, ``frame_step`` stride, ``2 * bins`` filters: the real parts of
    every bin, then the imaginary parts). The output is ``sqrt(re**2 + im**2)`` of shape
    ``(batch, frames, bins)``. The basis is a non-trainable weight, so a model's window and transform
    travel with its weights.

    Args:
        frame_length: Samples per frame.
        frame_step: Samples between frames.
        bins: Frequency bins.
        padding: Samples of reflect padding before the first and after the last sample.
    """

    def __init__(self, frame_length: int, frame_step: int, bins: int, padding: tuple[int, int] = (0, 0), **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.bins = bins
        self.padding = tuple(padding)

    def build(self, input_shape):
        self.basis = self.add_weight(
            name="basis", shape=(self.frame_length, 1, 2 * self.bins), initializer="zeros", trainable=False
        )

    def call(self, x):
        if any(self.padding):
            x = keras.ops.pad(x, [(0, 0), self.padding], mode="reflect")
        spectrum = keras.ops.conv(x[..., None], self.basis, strides=self.frame_step, padding="valid")
        real, imag = spectrum[..., : self.bins], spectrum[..., self.bins :]
        return keras.ops.sqrt(real * real + imag * imag)

    def compute_output_shape(self, input_shape):
        batch, samples = input_shape
        padded = None if samples is None else samples + sum(self.padding)
        frames = None if padded is None else (padded - self.frame_length) // self.frame_step + 1
        return batch, frames, self.bins

    def get_config(self):
        return {
            **super().get_config(),
            "frame_length": self.frame_length,
            "frame_step": self.frame_step,
            "bins": self.bins,
            "padding": self.padding,
        }
