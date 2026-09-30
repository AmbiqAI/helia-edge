"""Export manifest (``helia-edge/manifest@1``); importable without Keras."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .recipe import NAME, SHA256
from .result import EnvironmentRecord, TensorRecord
from .spec import ExportSpec, IODType, TensorRole

MANIFEST_SCHEMA = "helia-edge/manifest@1"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FileRecord(_Strict):
    """A file written next to the manifest; ``path`` is relative to the manifest."""

    path: str
    sha256: SHA256
    bytes: int


class TensorEntry(_Strict):
    """A model input or output; dynamic dimensions are -1."""

    name: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: IODType
    scale: float | None
    zero_point: int | None

    @classmethod
    def from_record(cls, record: TensorRecord) -> "TensorEntry":
        return cls(**vars(record))


class ReferenceRecord(_Strict):
    """Reference inputs as fed to the model and its outputs from LiteRT's reference kernels."""

    inputs: FileRecord
    outputs: FileRecord
    resolver: Literal["builtin_ref"] = "builtin_ref"


class EnvironmentEntry(_Strict):
    """Versions that can change exported bytes; ``verify`` refuses to compare across a difference."""

    helia_edge: str
    helia_edge_commit: str | None
    python: str
    platform: str
    packages: dict[str, str | None]

    @classmethod
    def from_record(cls, record: EnvironmentRecord) -> "EnvironmentEntry":
        return cls(
            helia_edge=record.helia_edge,
            helia_edge_commit=record.helia_edge_commit,
            python=record.python,
            platform=record.platform,
            packages=dict(record.packages),
        )


class ManifestEntry(_Strict):
    """One exported (or imported) model. ``spec`` is None for a ``tflite_import`` recipe."""

    name: NAME
    spec: ExportSpec | None
    model: FileRecord
    inputs: tuple[TensorEntry, ...]
    outputs: tuple[TensorEntry, ...]
    reference: ReferenceRecord | None = None


class ExportManifest(_Strict):
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
