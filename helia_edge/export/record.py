"""Export record (``helia-edge/export-record@1``): what ``export`` produced and how to produce it again.

One small JSON file per exported artifact, written next to it. With the weights it names by digest and
the calibration it names by sha256, it is enough to rebuild the model, export it again and compare the
bytes. It names files only by content hash and an optional URI, never by local path. Importable without
Keras.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator

from ..importers.mapping import SHA256
from ..models.spec import ModelSpec
from .result import EnvironmentRecord, HeliaEdgeSource, TensorRecord
from .spec import (
    CALIBRATED,
    VALID_IO,
    ConversionMode,
    IODType,
    Precision,
    TensorRole,
    check_dense_per_channel,
    check_resets,
)

RECORD_SCHEMA = "helia-edge/export-record@1"
GOLDEN_SCHEMA = "helia-model-zoo/golden@2"
DIGEST = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


_CONFIG = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Source(BaseModel):
    """A file named by content: its sha256 and, optionally, where to fetch it (never a local path)."""

    model_config = _CONFIG

    sha256: SHA256
    uri: str | None = None


class WeightImport(BaseModel):
    """Weights imported from another framework through ``mapping`` (a name in the family's ``MAPPINGS``)."""

    model_config = _CONFIG

    mapping: str
    source: Source


class WeightsRecord(BaseModel):
    """The weights, by ``weights_digest``, and their import when they came from another framework."""

    model_config = _CONFIG

    digest: DIGEST
    import_: WeightImport | None = Field(default=None, alias="import")


class ExportOptions(BaseModel):
    """Format options of a LiteRT export.

    Attributes:
        strict: For calibrated precisions, refuse operators without an integer kernel.
        state_tie_tolerance: For calibrated precisions with integer I/O, the largest relative difference
            between the scales of a state pair that export ties to one scale (see ``ExportSpec``).
        mode: How the model is traced: ``keras`` keeps the model's batch; ``concrete`` traces batch 1, so it
            exports ``batch_size`` 1 only.
        dense_per_channel: For calibrated precisions, quantize FULLY_CONNECTED weights per output channel
            (True, the converter's default) or with one scale per tensor (False). Convolutions stay per
            channel either way. The CMSIS-NN int16 FULLY_CONNECTED kernel of LiteRT for Microcontrollers
            accepts per-tensor weights only, so a16w8 exports for it use False; the reference and heliaRT
            kernels run either. Float precisions refuse False.
    """

    model_config = _CONFIG

    strict: StrictBool = True
    state_tie_tolerance: StrictFloat = Field(default=0.01, ge=0.0, le=0.5)
    mode: Literal[ConversionMode.KERAS, ConversionMode.CONCRETE] = ConversionMode.KERAS
    dense_per_channel: StrictBool = True


class CalibrationRecord(BaseModel):
    """The calibration array as stored (``.npy``), its length and, for a streaming model, the state resets."""

    model_config = _CONFIG

    sha256: SHA256
    uri: str | None = None
    samples: int = Field(gt=0)
    resets: tuple[int, ...] = ()


class ExportSettings(BaseModel):
    """How the artifact was exported. ``batch_size`` is the batch of every model input."""

    model_config = _CONFIG

    format: Literal["litert"] = "litert"
    precision: Precision
    io_dtype: IODType
    batch_size: StrictInt = Field(ge=1, le=2**31 - 1)  # LiteRT tensor dimensions are int32
    options: ExportOptions = ExportOptions()
    calibration: CalibrationRecord | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "ExportSettings":
        if self.io_dtype not in VALID_IO[self.precision]:
            raise ValueError(f"io_dtype {self.io_dtype} is not valid for precision {self.precision}")
        if self.options.mode is ConversionMode.CONCRETE and self.batch_size != 1:
            raise ValueError(f"concrete mode traces batch 1; it cannot export batch_size {self.batch_size}")
        if self.precision in CALIBRATED and self.batch_size != 1:
            raise ValueError(f"precision {self.precision} exports with batch_size 1, not {self.batch_size}")
        if (self.calibration is not None) != (self.precision in CALIBRATED):
            raise ValueError(
                f"precision {self.precision} {'needs' if self.precision in CALIBRATED else 'takes no'} calibration"
            )
        check_dense_per_channel(self.precision, self.options.dense_per_channel)
        if self.calibration is not None:
            resets = self.calibration.resets
            check_resets(resets, self.calibration.samples, stateful=True, what="Calibration")
        return self


class FileRecord(BaseModel):
    """A file written next to the record, by name, sha256 and size."""

    model_config = _CONFIG

    file: str
    sha256: SHA256
    bytes: int = Field(ge=0)


class TensorEntry(BaseModel):
    """A model input or output; dynamic dimensions are -1. A state tensor has the index ``pair`` of its
    state pair (``state_in_k`` and ``state_out_k``)."""

    model_config = _CONFIG

    name: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: IODType
    scale: float | None
    zero_point: int | None
    pair: int | None = None

    @classmethod
    def from_record(cls, record: TensorRecord) -> "TensorEntry":
        return cls(**vars(record))


class IORecord(BaseModel):
    """The artifact's inputs and outputs in subgraph order, read back from the artifact.

    ``state_scales_tied`` is True when every integer state pair has one scale and zero point, and None
    without integer state pairs.
    """

    model_config = _CONFIG

    inputs: tuple[TensorEntry, ...]
    outputs: tuple[TensorEntry, ...]
    state_scales_tied: bool | None = None


class GoldenRecord(BaseModel):
    """A ``helia-model-zoo/golden@2`` sequence: ``steps`` calls with the state carried and reset at
    ``resets``, run with LiteRT's reference kernels from the signal named by ``inputs``."""

    model_config = _CONFIG

    file: FileRecord
    schema_: Literal["helia-model-zoo/golden@2"] = Field(default=GOLDEN_SCHEMA, alias="schema")
    kind: Literal["sequence"] = "sequence"
    steps: int = Field(gt=0)
    resets: tuple[int, ...] = ()
    inputs: Source

    @model_validator(mode="after")
    def _resets_within_steps(self) -> "GoldenRecord":
        check_resets(self.resets, self.steps, stateful=True, what="Golden")
        return self


class HeliaEdgeRecord(BaseModel):
    """The helia-edge that exported: its version, how the version identifies the code, and the commit
    for a VCS install."""

    model_config = _CONFIG

    version: str
    source: HeliaEdgeSource
    commit: str | None = None


class EnvironmentEntry(BaseModel):
    """Versions that can change exported bytes."""

    model_config = _CONFIG

    helia_edge: HeliaEdgeRecord
    python: str
    platform: str
    packages: dict[str, str | None]

    @classmethod
    def from_record(cls, record: EnvironmentRecord) -> "EnvironmentEntry":
        return cls(
            helia_edge=HeliaEdgeRecord(
                version=record.helia_edge, source=record.helia_edge_source, commit=record.helia_edge_commit
            ),
            python=record.python,
            platform=record.platform,
            packages=dict(record.packages),
        )


class ExportRecord(BaseModel):
    """One exported artifact: the model, its weights, how it was exported, and the result.

    ``model``, ``weights`` and ``export`` together identify the export. ``model`` is None for a model
    that no ``ModelSpec`` describes; such a record cannot be rebuilt.
    """

    model_config = _CONFIG

    schema_: Literal["helia-edge/export-record@1"] = Field(default=RECORD_SCHEMA, alias="schema")
    model: ModelSpec | None
    weights: WeightsRecord
    export: ExportSettings
    artifact: FileRecord
    io: IORecord
    golden: GoldenRecord | None = None
    environment: EnvironmentEntry

    @model_validator(mode="after")
    def _artifact_has_the_batch(self) -> "ExportRecord":
        batches = {entry.shape[0] for entry in (*self.io.inputs, *self.io.outputs) if entry.shape}
        if batches - {self.export.batch_size}:
            raise ValueError(
                f"The artifact's input and output batch {sorted(batches)} is not batch_size {self.export.batch_size}"
            )
        return self

    def write(self, path: Path | str) -> None:
        """Write the record as JSON."""
        Path(path).write_text(self.model_dump_json(indent=1, by_alias=True) + "\n")

    @classmethod
    def read(cls, path: Path | str) -> "ExportRecord":
        """Read a record written by ``write``."""
        return cls.model_validate_json(Path(path).read_text())


def file_record(name: str, data: bytes) -> FileRecord:
    """The record of ``data`` written as ``name``."""
    return FileRecord(file=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def weights_digest(model) -> str:
    """``sha256:`` over the model's weights in ``model.weights`` order, independent of the file they came from.

    For each weight it hashes the NumPy dtype name and the shape, each followed by a NUL byte, then the
    C-order bytes of the value. Weight paths are not hashed: Keras numbers unnamed layers by how many it
    has built in the process, so paths can differ between two builds of one spec. The record's ``model``
    names the architecture.
    """
    import keras
    import numpy as np

    digest = hashlib.sha256()
    for weight in model.weights:
        value = np.asarray(keras.ops.convert_to_numpy(weight))  # ascontiguousarray would make a 0-d value (1,)
        for field in (value.dtype.name, ",".join(map(str, value.shape))):
            digest.update(field.encode() + b"\0")
        digest.update(value.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"
