# Copyright (c) 2020-present Silero Team
# Licensed under MIT; see licenses/silero-vad-mit.txt.
"""Silero VAD v6 (16 kHz) as a streaming Keras model with explicit LSTM state.

Follows snakers4/silero-vad v6.2.2 (commit 60b7ffa), ``silero_vad_16k_op15.onnx``. One call takes
576 samples, the last 64 samples of the previous call followed by 512 new ones (32 ms at 16 kHz), and
returns the speech probability. The LSTM state is carried as ``state_in_0``/``state_in_1`` (h, c),
fed from the previous call's ``state_out_0``/``state_out_1``; zeros (and zero context) start an
independent stream. The float model is exact: STFT magnitude with a square root, no quantization or
NPU rewrites. Weights come from ``helia_edge.importers`` with ``SILERO_VAD_V6_ONNX``
(``silero_vad_params``).
"""

import keras

from ..layers.spectral import StftMagnitude
from ..layers.streaming import StreamingLSTMCell, state_input, state_output
from .silero_vad_params import SileroVadParams

SAMPLES = 576
"""Samples per call: 64 of context and 512 new ones."""
UNITS = 128


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
    x = StftMagnitude(frame_length=256, frame_step=128, bins=129, padding=(0, 64), name="stft")(audio)
    for index, (filters, strides) in enumerate(((128, 1), (64, 2), (64, 2), (128, 1))):
        x = keras.layers.ZeroPadding1D(1, name=f"encoder{index}_pad")(x)
        x = keras.layers.Conv1D(filters, 3, strides=strides, activation="relu", name=f"encoder{index}")(x)
    # Four frames in leave one frame after the two strided convolutions: one LSTM step per call
    x = keras.layers.Reshape((UNITS,), name="frame")(x)
    h_next, c_next = StreamingLSTMCell(UNITS, name="lstm")([x, h, c])
    y = keras.layers.ReLU(name="decoder_relu")(h_next)
    prob = keras.layers.Dense(1, activation="sigmoid", name="prob")(y)
    outputs = [prob, state_output(0, h_next), state_output(1, c_next)]
    return keras.Model([audio, h, c], outputs, name=name or params.family)
