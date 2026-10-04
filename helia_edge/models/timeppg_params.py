"""Validated TimePPG architecture config and published channel presets; no backend imports."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

TIMEPPG_PRESET_SOURCE = "eml-eda/q-ppg@ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba"


class TimePPGParams(BaseModel):
    """TEMPONet-derived PPG heart-rate regressor (Burrello et al., 2022).

    Eleven widths in upstream order: nine convolution blocks (tcb00, tcb01,
    cb0, tcb10, tcb11, cb1, tcb20, tcb21, cb2) and two regressor layers
    (regr0, regr1). Dilations, kernels and strides are fixed by the
    architecture. No trained weights are published upstream.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["timeppg"] = "timeppg"
    channels: tuple[int, ...] = (26, 17, 42, 63, 41, 26, 30, 27, 16, 45, 80)

    @field_validator("channels")
    @classmethod
    def _eleven_positive_widths(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != 11 or any(width < 1 for width in value):
            raise ValueError("channels must be eleven positive widths")
        return value


# precision_search/model/TimePPG_float.py at TIMEPPG_PRESET_SOURCE.
TIMEPPG_PRESETS: Mapping[str, TimePPGParams] = {
    "timeppg_big": TimePPGParams(channels=(32, 32, 63, 64, 64, 121, 122, 104, 76, 82, 61)),
    "timeppg_medium": TimePPGParams(channels=(26, 17, 42, 63, 41, 26, 30, 27, 16, 45, 80)),
    "timeppg_small": TimePPGParams(channels=(2, 3, 2, 13, 2, 2, 31, 4, 9, 28, 77)),
}
