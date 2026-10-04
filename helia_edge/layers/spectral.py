"""
# Spectral Layers API

Classes:
    StftMagnitude: Magnitude of a short-time Fourier transform with a fixed, stored basis

Functions:
    max_projection: Magnitude of complex values without a square root, within 0.25%
"""

import keras
import numpy as np

from ..utils.export import helia_export

DIRECTIONS = 9


def max_projection(pairs):
    """Magnitude of complex values as the largest of 9 projections of (|re|, |im|), within 0.25%.

    The directions are spread from 0 to 90 degrees and scaled so the error is centred: the result is
    within -0.25% and +0.25% of ``sqrt(re**2 + im**2)``. It needs no square root, so integer kernels
    run it.

    Args:
        pairs: Real and imaginary parts on the last axis, ``(..., 2)``, rank 4.

    Returns:
        The magnitudes, ``(...)``.
    """
    theta = np.linspace(0, np.pi / 2, DIRECTIONS)
    gain = 2 / (1 + np.cos(np.pi / 4 / (DIRECTIONS - 1)))
    kernel = (gain * np.stack([np.cos(theta), np.sin(theta)]))[None, None]
    projected = keras.ops.conv(keras.ops.abs(pairs), keras.ops.cast(kernel, pairs.dtype), padding="valid")
    return keras.ops.max(projected, axis=-1)


@helia_export(path="helia_edge.layers.StftMagnitude")
class StftMagnitude(keras.layers.Layer):
    """Magnitude of a short-time Fourier transform computed as a strided convolution with a stored basis.

    The input ``(batch, samples)`` is reflect-padded by ``padding`` (start, end) samples and convolved
    with ``basis`` (``frame_length`` taps, ``frame_step`` stride, ``2 * bins`` filters: the real parts of
    every bin, then the imaginary parts). The output is the magnitude of shape ``(batch, frames, bins)``:
    ``sqrt(re**2 + im**2)``, or ``max_projection`` of (re, im). The basis is a non-trainable weight, so a
    model's window and transform travel with its weights.

    Args:
        frame_length: Samples per frame.
        frame_step: Samples between frames.
        bins: Frequency bins.
        padding: Samples of reflect padding before the first and after the last sample.
        magnitude: ``sqrt`` (exact) or ``max_projection`` (no square root, within 0.25%).
    """

    def __init__(
        self,
        frame_length: int,
        frame_step: int,
        bins: int,
        padding: tuple[int, int] = (0, 0),
        magnitude: str = "sqrt",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if magnitude not in ("sqrt", "max_projection"):
            raise ValueError(f"magnitude is 'sqrt' or 'max_projection', not {magnitude!r}")
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.bins = bins
        self.padding = tuple(padding)
        self.magnitude = magnitude

    def build(self, input_shape):
        self.basis = self.add_weight(
            name="basis", shape=(self.frame_length, 1, 2 * self.bins), initializer="zeros", trainable=False
        )

    def call(self, x):
        if any(self.padding):
            x = keras.ops.pad(x, [(0, 0), self.padding], mode="reflect")
        spectrum = keras.ops.conv(x[..., None], self.basis, strides=self.frame_step, padding="valid")
        real, imag = spectrum[..., : self.bins], spectrum[..., self.bins :]
        if self.magnitude == "max_projection":
            return max_projection(keras.ops.stack([real, imag], axis=-1))
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
            # Written only when not the default, so configs of the exact magnitude load in earlier versions
            **({"magnitude": self.magnitude} if self.magnitude != "sqrt" else {}),
        }
