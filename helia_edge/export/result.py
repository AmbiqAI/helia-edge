"""Export results and the environment that produced them; importable without Keras."""

import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import helia_edge

from .spec import ExportSpec, IODType, TensorRole

_VERSIONED = ("numpy", "keras", "tensorflow", "ai-edge-litert")


@dataclass(frozen=True)
class TensorRecord:
    """One model input or output. Dynamic dimensions are -1. I/O tensors are per-tensor quantized; float
    tensors have no scale."""

    name: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: IODType
    scale: float | None
    zero_point: int | None


@dataclass(frozen=True)
class EnvironmentRecord:
    """Versions that can change exported bytes.

    ``helia_edge`` is ``"unknown"`` when the imported package is not the installed distribution (for
    example a source tree on ``PYTHONPATH``). ``helia_edge_commit`` is set only for a distribution
    installed from a VCS URL.
    """

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


def _installed_helia_edge() -> tuple[str, str | None]:
    """Version and VCS commit of the installed distribution, if it is the imported package."""
    try:
        distribution = importlib.metadata.distribution("helia-edge")
    except importlib.metadata.PackageNotFoundError:
        return "unknown", None
    imported = Path(helia_edge.__file__).resolve().parent
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    url = direct_url.get("url", "")
    editable = direct_url.get("dir_info", {}).get("editable", False) and url.startswith("file://")
    installed = Path(url.removeprefix("file://")) / "helia_edge" if editable else distribution.locate_file("helia_edge")
    if Path(installed).resolve() != imported:
        return "unknown", None
    return distribution.version, direct_url.get("vcs_info", {}).get("commit_id")


def environment_record() -> EnvironmentRecord:
    """Describe the running environment."""
    version, commit = _installed_helia_edge()
    return EnvironmentRecord(
        helia_edge=version,
        helia_edge_commit=commit,
        python=platform.python_version(),
        packages=tuple((name, _version(name)) for name in _VERSIONED),
    )
