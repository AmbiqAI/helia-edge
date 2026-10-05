"""Reproduce an export from its record (``helia-edge export reproduce``) and create one from a spec file."""

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .record import EnvironmentEntry, ExportRecord, Source, WeightImport


@dataclass(frozen=True)
class Reproduction:
    """What ``reproduce`` found.

    Attributes:
        status: ``same`` (same artifact, I/O and golden), ``different``, ``environment`` (the environment
            differs and the comparison was not made) or ``input`` (a file is missing or does not match the
            record).
        differences: What differs, or which input is missing or does not match, one line each.
        environment: How the environment differs from the record's, or why it cannot export, one line each.
        skipped: What the record names but was not compared, one line each.
    """

    status: Literal["same", "different", "environment", "input"]
    differences: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()


def file_sha256(path: Path | str) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npy(path: Path | str) -> np.ndarray:
    """A ``.npy`` array, memory-mapped, so that a header claiming more data than the file holds is refused
    rather than allocated.

    Raises:
        ValueError: If the file is not a ``.npy`` array of the size its header states.
        OSError: If it cannot be read. EOFError: If it is empty.
    """
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except OverflowError as exc:  # a negative or overflowing dimension in the header
        raise ValueError(f"{path} has an invalid .npy header: {exc}") from exc
    if not isinstance(array, np.ndarray):  # an .npz archive
        array.close()
        raise ValueError(f"{path} is not a .npy array")
    return array


def load_spec(path: Path | str):
    """A ``ModelSpec`` from a YAML (``.yaml``/``.yml``) or JSON file.

    Raises:
        ValueError: If the file is not valid YAML or JSON, or not a valid ``ModelSpec``.
    """
    from ..models.spec import ModelSpec

    text = Path(path).read_text()
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(text)
        except (yaml.YAMLError, RecursionError) as exc:
            raise ValueError(f"{path} is not valid YAML: {exc}") from exc
        return ModelSpec.model_validate(data)
    return ModelSpec.model_validate_json(text)


def family_mapping(spec, name: str):
    """The weight mapping ``name`` of the spec's family (its params module's ``MAPPINGS``)."""
    mappings = getattr(sys.modules[type(spec.params).__module__], "MAPPINGS", {})
    if name not in mappings:
        raise ValueError(f"Family {spec.params.family!r} has no weight mapping {name!r}; it has {sorted(mappings)}")
    return mappings[name]


def environment_differences(recorded: EnvironmentEntry, current: EnvironmentEntry) -> list[str]:
    """How ``current`` differs from ``recorded``, in the versions that can change exported bytes."""
    differences = [
        f"helia_edge.{key}: {getattr(recorded.helia_edge, key)} -> {getattr(current.helia_edge, key)}"
        for key in ("version", "source", "commit")
        if getattr(recorded.helia_edge, key) != getattr(current.helia_edge, key)
    ]
    differences += [
        f"{key}: {getattr(recorded, key)} -> {getattr(current, key)}"
        for key in ("python", "platform")
        if getattr(recorded, key) != getattr(current, key)
    ]
    for package in sorted(set(recorded.packages) | set(current.packages)):
        if recorded.packages.get(package) != current.packages.get(package):
            differences.append(f"{package}: {recorded.packages.get(package)} -> {current.packages.get(package)}")
    return differences


def _load_samples(path: Path | str | None, what: str, sha256: str) -> np.ndarray:
    from .api import _array_sha256, _float32_samples

    if path is None:
        raise ValueError(f"The record names {what} (sha256 {sha256}); pass its .npy file")
    samples = _float32_samples(load_npy(path), what)
    if _array_sha256(samples) != sha256:
        raise ValueError(f"{path} as float32 has sha256 {_array_sha256(samples)}; the record names {sha256}")
    return samples


def litert_unavailable() -> tuple[str, ...]:
    """Why this process cannot export LiteRT, one line each; empty when it can."""
    try:
        import keras
    except ImportError as exc:
        return (f"Keras cannot be imported ({exc}); install helia-edge[litert]",)
    if keras.backend.backend() != "tensorflow":
        return (f"the Keras backend is {keras.backend.backend()!r}; LiteRT export needs KERAS_BACKEND=tensorflow",)
    return ()


def _files_beside(record_path: Path | str, record: ExportRecord) -> list[str]:
    """How the artifact and golden files beside the record differ from the record; absent files are not compared."""
    entries = [record.artifact] + ([record.golden.file] if record.golden is not None else [])
    differences = []
    for entry in entries:
        path = Path(record_path).parent / Path(entry.file).name
        if path.is_file() and (path.stat().st_size, file_sha256(path)) != (entry.bytes, entry.sha256):
            differences.append(
                f"{path.name} beside the record has {path.stat().st_size} bytes, sha256 {file_sha256(path)}; "
                f"the record names {entry.bytes} bytes, sha256 {entry.sha256}"
            )
    return differences


def reproduce(
    record_path: Path | str,
    weights: Path | str,
    calibration: Path | str | None = None,
    golden_inputs: Path | str | None = None,
    allow_env_mismatch: bool = False,
) -> Reproduction:
    """Rebuild the model a record describes, export it again with the same settings and compare.

    Args:
        record_path: The ``record.json``.
        weights: The ``.weights.h5`` file, or for imported weights the source file the mapping imports.
        calibration: The calibration ``.npy``, when the record has calibration.
        golden_inputs: The golden's input ``.npy``, to regenerate and compare the golden.
        allow_env_mismatch: Compare even when the environment differs from the record's.

    Returns:
        Reproduction: ``same`` when the weights digest, the export settings, the artifact, the I/O, the golden
        (with ``golden_inputs``) and the ``model.tflite`` and ``golden.npz`` beside the record (when present)
        match. A failure of the export itself is raised.
    """
    from ..importers import import_weights
    from .api import _reference_build, export, load_export_record
    from .result import environment_record

    try:
        record = ExportRecord.read(record_path)
    except (OSError, ValueError) as exc:
        return Reproduction("input", (f"{record_path} is not a readable export record: {exc}",))
    if record.model is None:
        return Reproduction("input", (f"{record_path} has no model spec, so the model cannot be rebuilt",))
    settings = record.export
    unused = [
        f"{flag} was given, but the record has no {what}"
        for flag, given, what in (
            ("--calibration", calibration, "calibration"),
            ("--golden-inputs", golden_inputs, "golden"),
        )
        if given is not None and (settings.calibration if what == "calibration" else record.golden) is None
    ]
    if unused:
        return Reproduction("input", tuple(unused))
    unavailable = litert_unavailable()
    if unavailable:
        return Reproduction("environment", environment=unavailable)
    environment = tuple(environment_differences(record.environment, EnvironmentEntry.from_record(environment_record())))
    if environment and not allow_env_mismatch:
        return Reproduction("environment", environment=environment)
    skipped = ("golden: pass --golden-inputs to compare it",) if record.golden and golden_inputs is None else ()
    try:
        if record.weights.import_ is not None:
            from ..models.spec import build

            imported = record.weights.import_
            sha256 = file_sha256(weights)
            if sha256 != imported.source.sha256:
                raise ValueError(f"{weights} has sha256 {sha256}; the record names {imported.source.sha256}")
            with _reference_build():
                model = build(record.model, batch_size=settings.batch_size)
            import_weights(model, family_mapping(record.model, imported.mapping), weights)
        else:
            model = load_export_record(record_path, weights)
        samples = (
            None
            if settings.calibration is None
            else _load_samples(calibration, "calibration", settings.calibration.sha256)
        )
        golden = None
        if record.golden is not None and golden_inputs is not None:
            golden = _load_samples(golden_inputs, "golden inputs", record.golden.inputs.sha256)
        differences = _files_beside(record_path, record)
    except (OSError, ValueError, EOFError) as exc:  # an input that cannot be read
        return Reproduction("input", (str(exc),), environment, skipped)
    again = export(
        model,
        precision=settings.precision,
        io_dtype=settings.io_dtype,
        calibration=samples,
        resets=() if settings.calibration is None else settings.calibration.resets,
        spec=record.model,
        batch_size=settings.batch_size,
        options=settings.options,
        weights_import=record.weights.import_,
        calibration_uri=None if settings.calibration is None else settings.calibration.uri,
    )
    if golden is not None and record.golden is not None:
        again = again.with_golden(golden, record.golden.resets, record.golden.inputs.uri)
    if again.record.weights != record.weights:
        differences.append(f"weights: {record.weights.digest} -> {again.record.weights.digest}")
    if again.record.export != record.export:
        differences.append(f"export settings: {record.export} -> {again.record.export}")
    if again.record.artifact != record.artifact:
        differences.append(f"artifact: {record.artifact.model_dump()} -> {again.record.artifact.model_dump()}")
    if again.record.io != record.io:
        differences.append("io: the inputs, outputs or state ties differ")
    if golden is not None and again.record.golden != record.golden:
        differences.append(f"golden: {record.golden} -> {again.record.golden}")
    return Reproduction("different" if differences else "same", tuple(differences), environment, skipped)


def imported_source(mapping, sha256: str) -> WeightImport:
    """The record of weights imported through ``mapping`` from a file with ``sha256``."""
    return WeightImport(mapping=mapping.name, source=Source(sha256=sha256, uri=mapping.source.uri))
