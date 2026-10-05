"""Export results and the environment that produced them; importable without Keras."""

import importlib.metadata
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import helia_edge

from .spec import ExportSpec, IODType, TensorRole

_VERSIONED = ("numpy", "keras", "tensorflow", "ai-edge-litert")

HeliaEdgeSource = Literal["release", "vcs", "local", "unknown"]


@dataclass(frozen=True)
class TensorRecord:
    """One model input or output. Dynamic dimensions are -1. I/O tensors are per-tensor quantized; float
    tensors have no scale. A state tensor (``role`` STATE) has the index ``pair`` of its state pair."""

    name: str
    role: TensorRole
    shape: tuple[int, ...]
    dtype: IODType
    scale: float | None
    zero_point: int | None
    pair: int | None = None


def _by_pair(records: Sequence[TensorRecord]) -> dict[int, TensorRecord]:
    pairs: dict[int, TensorRecord] = {}
    for record in records:
        if record.role == TensorRole.STATE:
            if record.pair is None:
                raise ValueError(f"State tensor {record.name!r} has no pair")
            pairs[record.pair] = record
    return pairs


def state_scales_tied(inputs: Sequence[TensorRecord], outputs: Sequence[TensorRecord]) -> bool | None:
    """Whether every integer state pair has one scale and zero point.

    Returns:
        bool | None: None when the model has no integer state pair, as for float I/O or a stateless model.

    Raises:
        ValueError: If a state tensor has no pair index, or a state input or output has no partner.
    """
    ins, outs = _by_pair(inputs), _by_pair(outputs)
    if ins.keys() != outs.keys():
        raise ValueError(f"State inputs {sorted(ins)} and outputs {sorted(outs)} do not pair up")
    pairs = [(ins[k], outs[k]) for k in sorted(ins) if ins[k].scale is not None or outs[k].scale is not None]
    if not pairs:
        return None
    return all((a.scale, a.zero_point) == (b.scale, b.zero_point) for a, b in pairs)


@dataclass(frozen=True)
class EnvironmentRecord:
    """Versions that can change exported bytes.

    ``helia_edge`` is ``"unknown"`` when the imported package is not the installed distribution (for
    example a source tree on ``PYTHONPATH``). ``helia_edge_commit`` is set only for a distribution
    installed from a VCS URL. ``helia_edge_source`` says how the version identifies the code: ``release``
    (installed from a package index), ``vcs`` (installed from a VCS URL at ``helia_edge_commit``),
    ``local`` (installed from a directory or an archive, or a source tree with egg-info metadata, which
    the version does not identify) or ``unknown``.
    """

    helia_edge: str
    helia_edge_commit: str | None
    python: str
    platform: str
    packages: tuple[tuple[str, str | None], ...]
    helia_edge_source: HeliaEdgeSource = "unknown"

    @property
    def identified(self) -> bool:
        """True when the recorded version or commit identifies the code that ran."""
        return self.helia_edge_source in ("release", "vcs")


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


def _installed_helia_edge() -> tuple[str, str | None, HeliaEdgeSource]:
    """Version, VCS commit and source of the installed distribution, if it is the imported package."""
    try:
        distribution = importlib.metadata.distribution("helia-edge")
    except importlib.metadata.PackageNotFoundError:
        return "unknown", None, "unknown"
    imported = Path(helia_edge.__file__).resolve().parent
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    url = direct_url.get("url", "")
    editable = direct_url.get("dir_info", {}).get("editable", False) and url.startswith("file://")
    installed = (
        Path(url.removeprefix("file://")) / "helia_edge"
        if editable
        else Path(str(distribution.locate_file("helia_edge")))
    )
    if Path(installed).resolve() != imported:
        return "unknown", None, "unknown"
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    # PEP 610: a distribution installed from an index has no direct_url.json. Installed distributions
    # have a RECORD; a source tree's .egg-info, which also has no direct_url.json, does not.
    installed_from_index = not direct_url and distribution.read_text("RECORD") is not None
    source = "vcs" if commit else ("release" if installed_from_index else "local")
    return distribution.version, commit, source


def environment_record() -> EnvironmentRecord:
    """Describe the running environment."""
    version, commit, source = _installed_helia_edge()
    return EnvironmentRecord(
        helia_edge=version,
        helia_edge_commit=commit,
        helia_edge_source=source,
        python=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        packages=tuple((name, _version(name)) for name in _VERSIONED),
    )


def unidentified_install(record: EnvironmentRecord) -> str:
    """Why ``record`` does not identify the helia-edge code, and how to install code that does."""
    return (
        f"helia-edge {record.helia_edge} ({record.helia_edge_source} install) does not identify its code; "
        "install a release, or install from git at a commit: "
        "uv pip install 'helia-edge @ git+https://github.com/AmbiqAI/helia-edge@<commit>'"
    )
