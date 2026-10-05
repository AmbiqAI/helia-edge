"""Typed export specification; importable without Keras or a training backend."""

import re
from collections.abc import Collection
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, model_validator


class Precision(StrEnum):
    """Numeric format of an exported graph.

    Attributes:
        FP32: float32 weights and compute.
        FP32_FP16W: float16 weight storage, dequantized at runtime; compute stays float32.
        FP16: native float16 graph; inputs, weights, activations and outputs are float16.
        A8W8: int8 activations and weights, calibrated.
        A16W8: int16 activations and int8 weights, calibrated.
    """

    FP32 = "fp32"
    FP32_FP16W = "fp32-w16"
    FP16 = "fp16"
    A8W8 = "a8w8"
    A16W8 = "a16w8"


class IODType(StrEnum):
    """Element type of an exported model's inputs and outputs."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT16 = "int16"


class ConversionMode(StrEnum):
    """How the Keras model is traced for conversion."""

    KERAS = "keras"
    SAVED_MODEL = "saved_model"
    CONCRETE = "concrete"


class TensorRole(StrEnum):
    """Role of a model input or output."""

    SIGNAL = "signal"
    STATE = "state"
    AUX = "aux"


_STATE_NAME = re.compile(r"state_(in|out)_(0|[1-9][0-9]*)")


def state_input_name(k: int) -> str:
    """Name of the input of state pair ``k``."""
    return f"state_in_{k}"


def state_output_name(k: int) -> str:
    """Name of the output of state pair ``k``."""
    return f"state_out_{k}"


def state_pair(name: str) -> tuple[str, int] | None:
    """``("in", k)`` or ``("out", k)`` for a state tensor name, otherwise None."""
    match = _STATE_NAME.fullmatch(name)
    return None if match is None else (match.group(1), int(match.group(2)))


# Legacy converter values map through this table only; values are never case-folded.
# Keys are QuantizationType values, so QuantizationType members look up directly.
LEGACY_PRECISION: dict[str, Precision] = {
    "FP32": Precision.FP32,
    "FP16": Precision.FP32_FP16W,
    "FP16_NATIVE": Precision.FP16,
    "INT8": Precision.A8W8,
    "INT16X8": Precision.A16W8,
}

# Legacy ConversionType values map through this table only.
LEGACY_MODE: dict[str, ConversionMode] = {
    "KERAS": ConversionMode.KERAS,
    "SAVED_MODEL": ConversionMode.SAVED_MODEL,
    "CONCRETE": ConversionMode.CONCRETE,
}

VALID_IO: dict[Precision, frozenset[IODType]] = {
    Precision.FP32: frozenset({IODType.FLOAT32}),
    Precision.FP32_FP16W: frozenset({IODType.FLOAT32}),
    Precision.FP16: frozenset({IODType.FLOAT16}),
    Precision.A8W8: frozenset({IODType.INT8, IODType.FLOAT32}),
    Precision.A16W8: frozenset({IODType.INT16, IODType.FLOAT32}),
}

CALIBRATED: frozenset[Precision] = frozenset({Precision.A8W8, Precision.A16W8})


class ExportSpec(BaseModel):
    """What to export. ``precision``, ``io_dtype`` and ``mode`` have no defaults.

    Attributes:
        format: Export format: ``litert``, a ``.tflite`` flatbuffer (the only one).
        precision: Numeric format of the graph.
        io_dtype: Input and output element type; must be valid for ``precision`` (``VALID_IO``).
        mode: How the model is traced for conversion.
        strict: For calibrated precisions, refuse operators without an integer kernel instead of
            falling back to float operators. Defaults to True; it does not affect float precisions.
        state_tie_tolerance: For calibrated precisions with integer I/O, export gives each state pair
            (``state_in_k``, ``state_out_k``) one scale and zero point covering both tensors' ranges, and
            refuses when that scale differs from either original by more than this fraction of it (at
            most 0.5). Models without state pairs ignore it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["litert"] = "litert"
    precision: Precision
    io_dtype: IODType
    mode: ConversionMode
    strict: StrictBool = True
    state_tie_tolerance: StrictFloat = Field(default=0.01, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def _io_dtype_matches_precision(self) -> "ExportSpec":
        valid = VALID_IO[self.precision]
        if self.io_dtype not in valid:
            allowed = ", ".join(sorted(valid))
            raise ValueError(
                f"io_dtype {self.io_dtype.value!r} is not valid for {self.precision.value!r}; use {allowed}"
            )
        return self


class BackendUnavailable(RuntimeError):
    """The active Keras backend cannot export: LiteRT export needs the TensorFlow backend."""


def check_resets(resets: Collection[int], steps: int, stateful: bool, what: str) -> None:
    """Refuse state resets that are not increasing steps from 1 to ``steps - 1`` of a streaming model.

    Raises:
        ValueError: If ``resets`` are given for a model without state, or are not increasing steps within
            the sequence (step 0 always starts from zero states).
    """
    resets = list(resets)
    if resets and not stateful:
        raise ValueError(f"{what} resets apply to streaming models only")
    if resets != sorted(set(resets)) or (resets and (resets[0] < 1 or resets[-1] >= steps)):
        bound = f"increasing steps between 1 and {steps - 1}" if steps > 1 else "absent for a single step"
        raise ValueError(f"{what} resets {resets} must be {bound}")
