"""Export manifest (``helia-edge/manifest@1``); importable without Keras."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .recipe import NAME, SHA256
from .result import EnvironmentRecord, HeliaEdgeSource, TensorRecord
from .spec import ExportSpec, IODType, TensorRole

MANIFEST_SCHEMA = "helia-edge/manifest@1"


class FileRecord(BaseModel):
    """A file written next to the manifest; ``path`` is relative to the manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: SHA256
    bytes: int


class TensorEntry(BaseModel):
    """A model input or output; dynamic dimensions are -1. A state tensor has the index ``pair`` of its
    state pair (``state_in_k`` and ``state_out_k``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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


GOLDEN_SCHEMA = "helia-model-zoo/golden@2"


class GoldenSource(BaseModel):
    """Where a golden's inputs came from: the recipe's reference array file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    sha256: SHA256


class GoldenRecord(BaseModel):
    """A ``helia-model-zoo/golden@2`` NPZ (``input_i``/``output_i`` in subgraph order, raw values).

    A ``sequence`` golden stacks ``steps`` calls along a leading axis, with each state output fed back as
    the next state input, and the state reset at the steps in ``resets``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    file: FileRecord
    schema_: Literal["helia-model-zoo/golden@2"] = Field(default=GOLDEN_SCHEMA, alias="schema")
    kind: Literal["sequence"]
    steps: int = Field(gt=0)
    resets: tuple[int, ...] = ()
    source: GoldenSource
    resolver: Literal["builtin_ref"] = "builtin_ref"


class ReferenceRecord(BaseModel):
    """Reference inputs as fed to the model and its outputs from LiteRT's reference kernels.

    A stateless model has ``inputs`` and ``outputs`` ``.npy`` files; a streaming model has a ``golden``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inputs: FileRecord | None = None
    outputs: FileRecord | None = None
    golden: GoldenRecord | None = None
    resolver: Literal["builtin_ref"] = "builtin_ref"

    @model_validator(mode="after")
    def _one_form(self) -> "ReferenceRecord":
        arrays = self.inputs is not None and self.outputs is not None
        if arrays == (self.golden is not None) or (self.inputs is None) != (self.outputs is None):
            raise ValueError("A reference has either inputs and outputs files or a golden")
        return self


class EnvironmentEntry(BaseModel):
    """Versions that can change exported bytes; ``verify`` refuses to compare across a difference.

    ``helia_edge_source`` is how the version identifies the code (see ``EnvironmentRecord``); None in
    manifests written before it was recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    helia_edge: str
    helia_edge_commit: str | None
    helia_edge_source: HeliaEdgeSource | None = None
    python: str
    platform: str
    packages: dict[str, str | None]

    @classmethod
    def from_record(cls, record: EnvironmentRecord) -> "EnvironmentEntry":
        return cls(
            helia_edge=record.helia_edge,
            helia_edge_commit=record.helia_edge_commit,
            helia_edge_source=record.helia_edge_source,
            python=record.python,
            platform=record.platform,
            packages=dict(record.packages),
        )


class ManifestEntry(BaseModel):
    """One exported (or imported) model. ``spec`` is None for a ``tflite_import`` recipe.

    ``state_scales_tied`` is True when every integer state pair has one scale and zero point, so a raw
    ``state_out_k`` fed back as ``state_in_k`` keeps its value; None without integer state pairs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: NAME
    spec: ExportSpec | None
    model: FileRecord
    inputs: tuple[TensorEntry, ...]
    outputs: tuple[TensorEntry, ...]
    reference: ReferenceRecord | None = None
    state_scales_tied: bool | None = None


class ExportManifest(BaseModel):
    """What ``helia-edge export run`` produced, with every file's sha256."""

    schema_: Literal["helia-edge/manifest@1"] = Field(alias="schema")
    recipe: FileRecord
    environment: EnvironmentEntry
    entries: tuple[ManifestEntry, ...]

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    def write(self, path: Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1, by_alias=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> "ExportManifest":
        return cls.model_validate_json(Path(path).read_text())
