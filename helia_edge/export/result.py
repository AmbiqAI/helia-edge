"""Export results and the environment that produced them; importable without Keras."""

import importlib.metadata
import json
import platform
from dataclasses import dataclass

from .spec import ExportSpec, IODType, TensorRole

_VERSIONED = ("numpy", "keras", "tensorflow", "ai-edge-litert")


@dataclass(frozen=True)
class TensorRecord:
    """One model input or output. I/O tensors are per-tensor quantized; float tensors have no scale."""

    name: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: IODType
    scale: float | None
    zero_point: int | None


@dataclass(frozen=True)
class EnvironmentRecord:
    """Versions that can change exported bytes. ``helia_edge_commit`` is None unless installed from VCS."""

    helia_edge: str
    helia_edge_commit: str | None
    python: str
    packages: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class ExportResult:
    """Exported model bytes with their identity."""

    spec: ExportSpec
    content: bytes
    sha256: str
    inputs: tuple[TensorRecord, ...]
    outputs: tuple[TensorRecord, ...]
    environment: EnvironmentRecord


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_record() -> EnvironmentRecord:
    """Describe the running environment."""
    commit = None
    try:
        direct_url = importlib.metadata.distribution("helia-edge").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return EnvironmentRecord(
        helia_edge=_version("helia-edge") or "unknown",
        helia_edge_commit=commit,
        python=platform.python_version(),
        packages=tuple((name, _version(name)) for name in _VERSIONED),
    )
