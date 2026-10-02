# Copyright (c) 2020-present Silero Team
# Licensed under MIT; see licenses/silero-vad-mit.txt.
"""Silero VAD v6 (16 kHz) as a streaming Keras model with explicit LSTM state.

Follows snakers4/silero-vad v6.2.2 (commit 60b7ffa), ``silero_vad_16k_op15.onnx``. One call takes
576 samples, the last 64 samples of the previous call followed by 512 new ones (32 ms at 16 kHz), and
returns the speech probability. The LSTM state is carried as ``state_in_0``/``state_in_1`` (h, c),
fed from the previous call's ``state_out_0``/``state_out_1``; zeros (and zero context) start an
independent stream. The float model is exact: STFT magnitude with a square root, no quantization or
NPU rewrites. Weights come from ``helia_edge.importers`` with ``SILERO_VAD_V6_ONNX``.
"""

import keras

from ..importers.mapping import Reshape, SourcePin, Transpose, WeightMapping, WeightRow
from ..layers.spectral import StftMagnitude
from ..layers.streaming import StreamingLSTMCell, state_input, state_output
from .silero_vad_params import SileroVadParams

SAMPLES = 576
"""Samples per call: 64 of context and 512 new ones."""
UNITS = 128


def silero_vad_v6(params: SileroVadParams | None = None) -> keras.Model:
    """Build the Silero VAD v6 16 kHz streaming model, untrained, with batch size 1.

    Inputs are ``audio`` (1, 576) float32 in [-1, 1] and the state ``state_in_0`` and ``state_in_1``
    (1, 128). Outputs are ``prob`` (1, 1) and ``state_out_0`` and ``state_out_1``.

    Args:
        params: The model's parameters; the defaults when None.
    """
    name = (params or SileroVadParams()).name
    audio = keras.Input((SAMPLES,), batch_size=1, name="audio")
    h, c = state_input(0, (UNITS,), batch_size=1), state_input(1, (UNITS,), batch_size=1)
    x = StftMagnitude(frame_length=256, frame_step=128, bins=129, padding=(0, 64), name="stft")(audio)
    for index, (filters, strides) in enumerate(((128, 1), (64, 2), (64, 2), (128, 1))):
        x = keras.layers.ZeroPadding1D(1, name=f"encoder{index}_pad")(x)
        x = keras.layers.Conv1D(filters, 3, strides=strides, activation="relu", name=f"encoder{index}")(x)
    # Four frames in leave one frame after the two strided convolutions: one LSTM step per call
    x = keras.layers.Reshape((UNITS,), name="frame")(x)
    h_next, c_next = StreamingLSTMCell(UNITS, name="lstm")([x, h, c])
    y = keras.layers.ReLU(name="decoder_relu")(h_next)
    prob = keras.layers.Dense(1, activation="sigmoid", name="prob")(y)
    return keras.Model([audio, h, c], [prob, state_output(0, h_next), state_output(1, c_next)], name=name)


def _conv(index: int) -> tuple[WeightRow, WeightRow]:
    prefix = f"model.encoder.{index}.reparam_conv"
    return (
        WeightRow(
            sources=(f"{prefix}.weight",),
            transforms=(Transpose(perm=(2, 1, 0)),),
            layer=f"encoder{index}",
            weight="kernel",
        ),
        WeightRow(sources=(f"{prefix}.bias",), layer=f"encoder{index}", weight="bias"),
    )


SILERO_VAD_V6_ONNX = WeightMapping(
    name="silero_vad_v6_onnx",
    format="onnx",
    source=SourcePin(
        uri="https://github.com/snakers4/silero-vad/raw/60b7ffa243625ebdc1070275a29f18c87843786a/src/silero_vad/data/silero_vad_16k_op15.onnx",
        sha256="7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49",
        note="v6.2.2, MIT. silero_vad_16k.safetensors in the same repository holds different weights.",
    ),
    rows=(
        # ONNX Conv weights are (out, in, k); Keras Conv1D kernels are (k, in, out)
        WeightRow(
            sources=("model.stft.forward_basis_buffer",),
            transforms=(Transpose(perm=(2, 1, 0)),),
            layer="stft",
            weight="basis",
        ),
        *_conv(0),
        *_conv(1),
        *_conv(2),
        *_conv(3),
        # PyTorch LSTM gates (i, f, g, o) are already in Keras order; the two biases add
        WeightRow(
            sources=("model.decoder.rnn.weight_ih",),
            transforms=(Transpose(perm=(1, 0)),),
            layer="lstm",
            weight="kernel",
        ),
        WeightRow(
            sources=("model.decoder.rnn.weight_hh",),
            transforms=(Transpose(perm=(1, 0)),),
            layer="lstm",
            weight="recurrent_kernel",
        ),
        WeightRow(
            sources=("model.decoder.rnn.bias_ih", "model.decoder.rnn.bias_hh"),
            combine="sum",
            layer="lstm",
            weight="bias",
        ),
        WeightRow(
            sources=("model.decoder.decoder.2.weight",),
            transforms=(Reshape(shape=(UNITS, 1)),),
            layer="prob",
            weight="kernel",
        ),
        WeightRow(sources=("model.decoder.decoder.2.bias",), layer="prob", weight="bias"),
    ),
)
"""Mapping of every weight of ``silero_vad_v6`` from the pinned v6.2.2 ONNX file."""
