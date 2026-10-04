"""Typed parameters of the Silero VAD v6 streaming model; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SileroVadParams(BaseModel):
    """Silero VAD v6 (16 kHz). Every value is fixed by the v6.2.2 weights.

    Attributes:
        family: Model family.
        sample_rate: Audio sample rate in Hz.
        context: Samples of the previous call repeated at the start of each call.
        hop: New samples per call (32 ms).
        units: LSTM state size of ``state_in_0``/``state_in_1`` (h, c).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["silero_vad"] = "silero_vad"
    sample_rate: Literal[16000] = 16000
    context: Literal[64] = 64
    hop: Literal[512] = 512
    units: Literal[128] = 128

    @property
    def samples(self) -> int:
        """Samples per call: ``context + hop``."""
        return self.context + self.hop

    @property
    def input_shape(self) -> tuple[int, ...]:
        """The fixed audio shape ``(samples,)``, without the batch axis."""
        return (self.samples,)
