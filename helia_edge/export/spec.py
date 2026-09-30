"""Typed export specification; importable without Keras or a training backend."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator


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
        format: Export format. ``litert`` (a ``.tflite`` flatbuffer), the default, is built in;
            other formats come from exporters registered in ``helia_edge.registry.exporters``.
        precision: Numeric format of the graph.
        io_dtype: Input and output element type; must be valid for ``precision`` (``VALID_IO``).
        mode: How the model is traced for conversion.
        strict: For calibrated precisions, refuse operators without an integer kernel instead of
            falling back to float operators. Defaults to True; it does not affect float precisions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: str = "litert"
    precision: Precision
    io_dtype: IODType
    mode: ConversionMode
    strict: StrictBool = True

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
    """The active Keras backend has no exporter for the requested format."""
