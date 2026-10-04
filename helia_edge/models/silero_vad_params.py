"""Typed parameters of the Silero VAD v6 streaming model and its weight mapping; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..importers.mapping import Reshape, SourcePin, Transpose, WeightMapping, WeightRow


class SileroVadParams(BaseModel):
    """Silero VAD v6 (16 kHz). The geometry is fixed by the v6.2.2 weights; the options choose how the
    model computes it, and every option keeps the same weights.

    Attributes:
        family: Model family.
        sample_rate: Audio sample rate in Hz.
        context: Samples of the previous call repeated at the start of each call.
        hop: New samples per call (32 ms).
        units: LSTM state size of ``state_in_0``/``state_in_1`` (h, c).
        stft: ``conv1d`` frames the reflect-padded audio with a strided convolution. ``conv_blocks``
            convolves 64-sample blocks instead, with the right reflect padding folded into the last
            frame's kernel; it is exact in float.
        magnitude: ``sqrt`` is the exact magnitude of each bin. ``max_projection`` is the largest of 9
            projections of (|re|, |im|) onto directions from 0 to 90 degrees, within +-0.24%; it needs no
            square root.
        encoder_tail: ``conv`` runs the last two encoder layers as convolutions. ``live_taps`` runs them as
            dense layers over the kernel taps that see real frames rather than padding; it is exact in float.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["silero_vad"] = "silero_vad"
    sample_rate: Literal[16000] = 16000
    context: Literal[64] = 64
    hop: Literal[512] = 512
    units: Literal[128] = 128
    stft: Literal["conv1d", "conv_blocks"] = "conv1d"
    magnitude: Literal["sqrt", "max_projection"] = "sqrt"
    encoder_tail: Literal["conv", "live_taps"] = "conv"

    @property
    def samples(self) -> int:
        """Samples per call: ``context + hop``."""
        return self.context + self.hop

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The fixed audio shape ``(samples,)``, without the batch axis."""
        return (self.samples,)


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
    source=SourcePin(
        uri="https://github.com/snakers4/silero-vad/raw/60b7ffa243625ebdc1070275a29f18c87843786a/src/silero_vad/data/silero_vad_16k_op15.onnx",
        sha256="7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49",
        format="onnx",
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
            source_shape=(1, SileroVadParams().units, 1),
            transforms=(Reshape(shape=(SileroVadParams().units, 1)),),
            layer="prob",
            weight="kernel",
        ),
        WeightRow(sources=("model.decoder.decoder.2.bias",), layer="prob", weight="bias"),
    ),
)
"""Mapping of every weight of the Silero VAD v6 model (``build(SileroVadParams())``) from the pinned v6.2.2
ONNX file."""

MAPPINGS: dict[str, WeightMapping] = {SILERO_VAD_V6_ONNX.name: SILERO_VAD_V6_ONNX}
"""Weight mappings for this family, by name."""
