"""Export record (``helia-edge/export-record@1``): what ``export`` produced and how to produce it again.

One small JSON file per exported artifact, written next to it. With the weights it names by digest and
the calibration it names by sha256, it is enough to rebuild the model, export it again and compare the
bytes. It names files only by content hash and an optional URI, never by local path. Importable without
Keras.
"""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..importers.mapping import SHA256
from ..models.spec import ModelSpec
from .result import EnvironmentRecord, HeliaEdgeSource, TensorRecord
from .spec import ConversionMode, IODType, Precision, TensorRole

RECORD_SCHEMA = "helia-edge/export-record@1"
GOLDEN_SCHEMA = "helia-model-zoo/golden@2"
DIGEST = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Source(_Record):
    """A file named by content: its sha256 and, optionally, where to fetch it (never a local path)."""

    sha256: SHA256
    uri: str | None = None


class WeightImport(_Record):
    """Weights imported from another framework through ``mapping`` (a name in the family's ``MAPPINGS``)."""

    mapping: str
    source: Source


class WeightsRecord(_Record):
    """The weights, by ``weights_digest``, and their import when they came from another framework."""

    digest: DIGEST
    import_: WeightImport | None = Field(default=None, alias="import")


class ExportOptions(_Record):
    """Format options of a LiteRT export.

    Attributes:
        strict: For calibrated precisions, refuse operators without an integer kernel.
        state_tie_tolerance: For calibrated precisions with integer I/O, the largest relative difference
            between the scales of a state pair that export ties to one scale (see ``ExportSpec``).
        mode: How the model is traced: ``keras`` or ``concrete``, both of which keep the model's batch.
    """

    strict: bool = True
    state_tie_tolerance: float = Field(default=0.01, ge=0.0, le=0.5)
    mode: Literal[ConversionMode.KERAS, ConversionMode.CONCRETE] = ConversionMode.KERAS


class CalibrationRecord(_Record):
    """The calibration array as stored (``.npy``), its length and, for a streaming model, the state resets."""

    sha256: SHA256
    uri: str | None = None
    samples: int = Field(gt=0)
    resets: tuple[int, ...] = ()


class ExportSettings(_Record):
    """How the artifact was exported. ``batch_size`` is the batch of every model input."""

    format: Literal["litert"] = "litert"
    precision: Literal[Precision.FP32, Precision.FP16, Precision.A8W8, Precision.A16W8]
    io_dtype: IODType
    batch_size: int = Field(ge=1)
    options: ExportOptions = ExportOptions()
    calibration: CalibrationRecord | None = None


class FileRecord(_Record):
    """A file written next to the record, by name, sha256 and size."""

    file: str
    sha256: SHA256
    bytes: int = Field(ge=0)


class TensorEntry(_Record):
    """A model input or output; dynamic dimensions are -1. A state tensor has the index ``pair`` of its
    state pair (``state_in_k`` and ``state_out_k``)."""

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


class IORecord(_Record):
    """The artifact's inputs and outputs in subgraph order, read back from the artifact.

    ``state_scales_tied`` is True when every integer state pair has one scale and zero point, and None
    without integer state pairs.
    """

    inputs: tuple[TensorEntry, ...]
    outputs: tuple[TensorEntry, ...]
    state_scales_tied: bool | None = None


class GoldenRecord(_Record):
    """A ``helia-model-zoo/golden@2`` sequence: ``steps`` calls with the state carried and reset at
    ``resets``, run with LiteRT's reference kernels from the signal named by ``inputs``."""

    file: FileRecord
    schema_: Literal["helia-model-zoo/golden@2"] = Field(default=GOLDEN_SCHEMA, alias="schema")
    kind: Literal["sequence"] = "sequence"
    steps: int = Field(gt=0)
    resets: tuple[int, ...] = ()
    inputs: Source


class HeliaEdgeRecord(_Record):
    """The helia-edge that exported: its version, how the version identifies the code, and the commit
    for a VCS install."""

    version: str
    source: HeliaEdgeSource
    commit: str | None = None


class EnvironmentEntry(_Record):
    """Versions that can change exported bytes."""

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


class ExportRecord(_Record):
    """One exported artifact: the model, its weights, how it was exported, and the result.

    ``model``, ``weights`` and ``export`` together identify the export. ``model`` is None for a model
    that no ``ModelSpec`` describes; such a record cannot be rebuilt.
    """

    schema_: Literal["helia-edge/export-record@1"] = Field(default=RECORD_SCHEMA, alias="schema")
    model: ModelSpec | None
    weights: WeightsRecord
    export: ExportSettings
    artifact: FileRecord
    io: IORecord
    golden: GoldenRecord | None = None
    environment: EnvironmentEntry

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

    For each weight it hashes the UTF-8 path, the NumPy dtype name and the shape, each followed by a NUL
    byte, then the C-order bytes of the value.
    """
    import keras
    import numpy as np

    digest = hashlib.sha256()
    for weight in model.weights:
        value = np.ascontiguousarray(keras.ops.convert_to_numpy(weight))
        for field in (weight.path, value.dtype.name, ",".join(map(str, value.shape))):
            digest.update(field.encode() + b"\0")
        digest.update(value.tobytes())
    return f"sha256:{digest.hexdigest()}"
