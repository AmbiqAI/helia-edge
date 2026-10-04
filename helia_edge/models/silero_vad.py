# Copyright (c) 2020-present Silero Team
# Licensed under MIT; see licenses/silero-vad-mit.txt.
"""Silero VAD v6 (16 kHz) as a streaming Keras model with explicit LSTM state.

Follows snakers4/silero-vad v6.2.2 (commit 60b7ffa), ``silero_vad_16k_op15.onnx``. One call takes
576 samples, the last 64 samples of the previous call followed by 512 new ones (32 ms at 16 kHz), and
returns the speech probability. The LSTM state is carried as ``state_in_0``/``state_in_1`` (h, c),
fed from the previous call's ``state_out_0``/``state_out_1``; zeros (and zero context) start an
independent stream. Weights come from ``helia_edge.importers`` with ``SILERO_VAD_V6_ONNX``
(``silero_vad_params``).

The default options compute the reference model exactly: an STFT magnitude with a square root. The
other options (``SileroVadParams``) compute the same model from operators that integer kernels and
NPUs run, deriving what they need (block kernels, the folded padding, tap slices) from the same
weights inside the graph, so an exporter folds them into constants.
"""

import keras
import numpy as np

from ..layers.streaming import StreamingLSTMCell, state_input, state_output
from ..utils.export import helia_export
from .silero_vad_params import SileroVadParams

SAMPLES = 576
"""Samples per call: 64 of context and 512 new ones."""
UNITS = 128
FRAME, STEP, BINS, BLOCK, DIRECTIONS = 256, 128, 129, 64, 9


def _projection_kernel() -> np.ndarray:
    """(1, 1, 2, 9) unit directions from 0 to 90 degrees, scaled so the error is centred within +-0.24%."""
    theta = np.linspace(0, np.pi / 2, DIRECTIONS)
    gain = 2 / (1 + np.cos(np.pi / 4 / (DIRECTIONS - 1)))
    return (gain * np.stack([np.cos(theta), np.sin(theta)]))[None, None].astype(np.float32)


@helia_export()
class SileroStft(keras.layers.Layer):
    """STFT magnitude of one call from the stored basis (256, 1, 258).

    The basis holds the real parts of every bin, then the imaginary parts, and frames are reflect-padded
    by 64 samples at the end. ``stft`` and ``magnitude`` are the ``SileroVadParams`` options. The output
    is (batch, 4, 129) with ``conv1d``; with ``conv_blocks`` it is (batch, 4, 1, 129), the frames as rows
    of an image, which the encoder convolves without reshapes.
    """

    def __init__(self, stft: str = "conv1d", magnitude: str = "sqrt", **kwargs):
        super().__init__(**kwargs)
        self.stft = stft
        self.magnitude = magnitude

    def build(self, input_shape):
        self.basis = self.add_weight(name="basis", shape=(FRAME, 1, 2 * BINS), initializer="zeros", trainable=False)

    def _blocks(self, audio):
        """(batch, 4, 129, 2) real and imaginary parts, from convolutions over 64-sample blocks."""
        taps = keras.ops.take(self.basis[:, 0, :], np.stack([np.arange(BINS), np.arange(BINS) + BINS], 1).ravel(), 1)
        blocks = keras.ops.reshape(audio, (-1, SAMPLES // BLOCK, 1, BLOCK))
        main = keras.ops.reshape(taps, (FRAME // BLOCK, 1, BLOCK, 2 * BINS))
        first = keras.ops.conv(blocks, main, strides=(2, 1), padding="valid")  # frames 0 to 2
        # Frame 3 reads samples 384..639; samples 576..639 reflect samples 574 down to 511, so their taps
        # add to the taps of rows 190 down to 127 (a concatenation, which exporters fold; a pad is not)
        reflected = taps[127:191] + keras.ops.flip(taps[3 * BLOCK :], 0)
        folded = keras.ops.concatenate([taps[:127], reflected, taps[191 : 3 * BLOCK]], axis=0)
        last = keras.ops.conv(
            blocks[:, 6 : SAMPLES // BLOCK], keras.ops.reshape(folded, (3, 1, BLOCK, 2 * BINS)), padding="valid"
        )
        return keras.ops.reshape(keras.ops.concatenate([first, last], axis=1), (-1, 4, BINS, 2))

    def call(self, audio):
        if self.stft == "conv_blocks":
            pairs = self._blocks(audio)
            if self.magnitude == "max_projection":
                projected = keras.ops.conv(keras.ops.abs(pairs), _projection_kernel(), padding="valid")
                magnitude = keras.ops.max(projected, axis=-1)
            else:
                magnitude = keras.ops.sqrt(keras.ops.sum(pairs * pairs, axis=-1))
            return keras.ops.reshape(magnitude, (-1, 4, 1, BINS))
        x = keras.ops.pad(audio, [(0, 0), (0, BLOCK)], mode="reflect")
        spectrum = keras.ops.conv(x[..., None], self.basis, strides=STEP, padding="valid")
        real, imag = spectrum[..., :BINS], spectrum[..., BINS:]
        if self.magnitude == "max_projection":
            pairs = keras.ops.stack([real, imag], axis=-1)
            projected = keras.ops.conv(keras.ops.abs(pairs), _projection_kernel(), padding="valid")
            return keras.ops.max(projected, axis=-1)
        return keras.ops.sqrt(real * real + imag * imag)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 1, BINS) if self.stft == "conv_blocks" else (input_shape[0], 4, BINS)

    def get_config(self):
        return {**super().get_config(), "stft": self.stft, "magnitude": self.magnitude}


@helia_export()
class SileroFrameConv(keras.layers.Layer):
    """An encoder ``Conv1D`` with ReLU over frames held as rows of a (frames, 1, channels) image.

    The kernel keeps the ``Conv1D`` shape (3, channels, filters); input padding is a separate layer.
    """

    def __init__(self, filters: int, strides: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=(3, input_shape[-1], self.filters), initializer="glorot_uniform"
        )
        self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros")

    def call(self, x):
        y = keras.ops.conv(x, self.kernel[:, None], strides=(self.strides, 1), padding="valid")
        return keras.ops.relu(y + self.bias)

    def compute_output_shape(self, input_shape):
        frames = None if input_shape[1] is None else (input_shape[1] - 3) // self.strides + 1
        return (input_shape[0], frames, 1, self.filters)

    def get_config(self):
        return {**super().get_config(), "filters": self.filters, "strides": self.strides}


@helia_export()
class SileroLiveTaps(keras.layers.Layer):
    """A zero-padded encoder convolution as a dense layer over the kernel taps that see real frames.

    The kernel keeps the convolution's shape (3, channels, filters). ``taps`` lists the taps that see
    the input's frames, in order; the input is those frames' channels, flattened.
    """

    def __init__(self, filters: int, taps: tuple[int, ...], **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.taps = tuple(taps)

    def build(self, input_shape):
        channels = input_shape[-1] // len(self.taps)
        self.kernel = self.add_weight(name="kernel", shape=(3, channels, self.filters), initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros")

    def call(self, x):
        kernel = keras.ops.concatenate([self.kernel[tap] for tap in self.taps], axis=0)
        return keras.ops.relu(keras.ops.matmul(x, kernel) + self.bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.filters)

    def get_config(self):
        return {**super().get_config(), "filters": self.filters, "taps": self.taps}


def build(
    params: SileroVadParams,
    input_shape: tuple[int | None, ...] | None = None,
    *,
    batch_size: int | None = None,
    name: str | None = None,
) -> keras.Model:
    """Build the Silero VAD v6 16 kHz streaming model, untrained.

    Inputs are ``audio`` (576,) float32 in [-1, 1] and the state ``state_in_0`` and ``state_in_1``
    (128,). Outputs are ``prob`` (1,) and ``state_out_0`` and ``state_out_1``. Streaming and export use
    ``batch_size=1``.

    Args:
        params (SileroVadParams): Model parameters (all fixed by the v6.2.2 weights).
        input_shape (tuple[int | None, ...] | None): None, or the audio shape ``(576,)``.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``silero_vad`` unless ``name`` is given.
    """
    if input_shape is not None and tuple(input_shape) != (params.samples,):
        raise ValueError(f"Silero VAD takes audio shape ({params.samples},), not {tuple(input_shape)}")
    audio = keras.Input((params.samples,), batch_size=batch_size, name="audio")
    h, c = state_input(0, (UNITS,), batch_size=batch_size), state_input(1, (UNITS,), batch_size=batch_size)
    x = SileroStft(params.stft, params.magnitude, name="stft")(audio)
    layers = ((128, 1), (64, 2)) if params.encoder_tail == "live_taps" else ((128, 1), (64, 2), (64, 2), (128, 1))
    for index, (filters, strides) in enumerate(layers):
        if params.stft == "conv_blocks":
            x = keras.layers.ZeroPadding2D(((1, 1), (0, 0)), name=f"encoder{index}_pad")(x)
            x = SileroFrameConv(filters, strides, name=f"encoder{index}")(x)
        else:
            x = keras.layers.ZeroPadding1D(1, name=f"encoder{index}_pad")(x)
            x = keras.layers.Conv1D(filters, 3, strides=strides, activation="relu", name=f"encoder{index}")(x)
    if params.encoder_tail == "live_taps":
        # Two frames reach encoder2, whose padded stride-2 window sees them through taps 1 and 2; one frame
        # reaches encoder3, seen through its centre tap
        x = keras.layers.Reshape((UNITS,), name="frames")(x)
        x = SileroLiveTaps(64, (1, 2), name="encoder2")(x)
        x = SileroLiveTaps(UNITS, (1,), name="encoder3")(x)
    else:
        # Four frames in leave one frame after the two strided convolutions: one LSTM step per call
        x = keras.layers.Reshape((UNITS,), name="frame")(x)
    h_next, c_next = StreamingLSTMCell(UNITS, name="lstm")([x, h, c])
    y = keras.layers.ReLU(name="decoder_relu")(h_next)
    prob = keras.layers.Dense(1, activation="sigmoid", name="prob")(y)
    outputs = [prob, state_output(0, h_next), state_output(1, c_next)]
    return keras.Model([audio, h, c], outputs, name=name or params.family)
