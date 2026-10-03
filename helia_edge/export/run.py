"""Run an export recipe into a directory with a manifest, and verify a manifest by regenerating it."""

import hashlib
import os
import tempfile
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .manifest import (
    MANIFEST_SCHEMA,
    EnvironmentEntry,
    ExportManifest,
    FileRecord,
    GoldenRecord,
    GoldenSource,
    ManifestEntry,
    ReferenceRecord,
    TensorEntry,
)
from .recipe import (
    ArraySource,
    KerasFile,
    ParamsImport,
    ParamsSeed,
    ParamsWeights,
    PathSource,
    TfliteImport,
    UrlSource,
    load_recipe,
)
from .result import EnvironmentRecord, environment_record, state_scales_tied
from .spec import CALIBRATED, ExportSpec, state_pair


class SourceError(ValueError):
    """A recipe source is missing or does not match its sha256."""


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_dir() -> Path:
    return Path(os.environ.get("HELIA_EDGE_CACHE", Path.home() / ".cache" / "helia-edge")) / "sources"


def fetch(source: UrlSource | PathSource, base_dir: Path) -> Path:
    """Return a local path to the source file after checking its sha256.

    Path sources are relative to ``base_dir`` unless absolute. URL sources are downloaded once
    into ``$HELIA_EDGE_CACHE/sources`` (default ``~/.cache/helia-edge/sources``), named by sha256.
    """
    if isinstance(source, PathSource):
        path = Path(source.path)
        path = path if path.is_absolute() else base_dir / path
        if not path.is_file():
            raise SourceError(f"Source file not found: {path}")
    else:
        path = _cache_dir() / source.sha256
        if not path.is_file() or sha256_file(path) != source.sha256:
            from ..utils.file import download_file

            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            download_file(source.url, partial, progress=False)
            partial.replace(path)
    digest = sha256_file(path)
    if digest != source.sha256:
        raise SourceError(f"{path} has sha256 {digest}, expected {source.sha256}")
    return path


def load_array(source: ArraySource, base_dir: Path, samples: int | None = None) -> np.ndarray:
    """Load a ``.npy`` array, or key ``source.key`` of a ``.npz``; keep the first ``samples`` rows."""
    path = fetch(source.file, base_dir)
    location = source.file.path if isinstance(source.file, PathSource) else source.file.url
    with open(path, "rb") as f:
        archive = f.read(4) == b"PK\x03\x04"  # .npz files are zip archives; .npy files start with \x93NUMPY
    if archive:
        if source.key is None:
            raise SourceError(f"{location} is an .npz archive; set key")
        with np.load(path, allow_pickle=False) as arrays:
            array = arrays[source.key]
    else:
        if source.key is not None:
            raise SourceError(f"{location} is not an .npz archive; omit key")
        array = np.load(path, allow_pickle=False)
    if samples is not None and samples > len(array):
        raise SourceError(f"{location} has {len(array)} rows; the recipe asks for {samples}")
    return array if samples is None else array[:samples]


def _batch1(model):
    """Rebuild a single-input functional model whose batch dimension is not fixed to 1.

    A model with several inputs, such as a streaming model, must already have batch size 1.
    """
    import keras

    if len(model.inputs) != 1:
        if not any(state_pair(tensor.name) for tensor in model.inputs):
            raise ValueError(
                f"Recipes support single-input models and streaming models with state_in_k inputs; this model has "
                f"{len(model.inputs)} inputs"
            )
        if all(tensor.shape[0] == 1 for tensor in model.inputs):
            return model
        raise ValueError(f"A recipe model with {len(model.inputs)} inputs must have batch size 1 on every input")
    if model.input_shape[0] == 1:
        return model
    inputs = keras.Input(model.input_shape[1:], batch_size=1)
    return keras.Model(inputs, model(inputs), name=model.name)


def _signal_input(model) -> str | None:
    """The one input that is not a state input of a streaming model, or None for a stateless model."""
    names = [tensor.name for tensor in model.inputs]
    signals = [name for name in names if state_pair(name) is None]
    if len(signals) == len(names):
        return None
    if len(signals) != 1:
        raise ValueError(f"A recipe streaming model needs exactly one input that is not a state; got {signals}")
    return signals[0]


def _check_resets(resets, steps: int, stateful: bool, what: str) -> None:
    if resets and not stateful:
        raise ValueError(f"{what} resets apply to streaming models only")
    if list(resets) != sorted(set(resets)) or (resets and resets[-1] >= steps):
        bound = f"increasing steps between 1 and {steps - 1}" if steps > 1 else "absent for a single step"
        raise ValueError(f"{what} resets {list(resets)} must be {bound}")


def build_model(source: ParamsSeed | ParamsWeights | ParamsImport | KerasFile, base_dir: Path):
    """Build or load the recipe's Keras model with batch size 1, in a fresh Keras session.

    Auto-generated layer names become tensor names in the exported bytes and come from
    process-wide counters, so ``keras.backend.clear_session()`` runs first. This discards models
    built earlier in the process.
    """
    import keras

    from .architectures import resolve_architecture

    keras.backend.clear_session()

    if isinstance(source, ParamsSeed):
        keras.utils.set_random_seed(source.seed)
        model = resolve_architecture(source.architecture)(source.params, source.input_shape, source.num_classes)
    elif isinstance(source, ParamsWeights):
        model = resolve_architecture(source.architecture)(source.params, source.input_shape, source.num_classes)
        model.load_weights(fetch(source.weights, base_dir))
    elif isinstance(source, ParamsImport):
        from ..importers import import_weights
        from ..registry import weight_mappings

        mapping = weight_mappings.get(source.mapping)
        if source.weights.sha256 != mapping.source.sha256:
            raise SourceError(
                f"Weights sha256 {source.weights.sha256} is not the file mapping {source.mapping!r} is pinned to "
                f"({mapping.source.sha256}, {mapping.source.uri})"
            )
        model = resolve_architecture(source.architecture)(source.params, source.input_shape, source.num_classes)
        import_weights(model, mapping, fetch(source.weights, base_dir))
    else:
        from .. import register_keras_serializables

        register_keras_serializables()
        model = keras.models.load_model(fetch(source.file, base_dir), compile=False)
    return _batch1(model)


def _write(path: Path, data: bytes, root: Path) -> FileRecord:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return FileRecord(path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def _npy(array: np.ndarray) -> bytes:
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _golden(content: bytes, frames: np.ndarray, resets, source: GoldenSource, path: Path, out: Path) -> GoldenRecord:
    """Write a golden@2 sequence for a streaming model: ``frames`` are consecutive calls of its signal input."""
    import io

    from .runner import LiteRTStreamRunner

    runner = LiteRTStreamRunner(content, reference_kernels=True)
    if len(runner.signals) != 1:
        raise ValueError(
            f"A streaming reference needs exactly one input that is not a state; got {list(runner.signals)}"
        )
    (signal,) = runner.signals
    encoded = runner.encode(signal, frames.reshape(len(frames), *runner.inputs[signal]["shape"]))
    fed, produced = runner.run({signal: encoded}, resets)
    by_index = {details["index"]: name for name, details in runner.inputs.items()}
    out_by_index = {details["index"]: name for name, details in runner.outputs.items()}
    order_in = [by_index[d["index"]] for d in runner.interpreter.get_input_details()]
    order_out = [out_by_index[d["index"]] for d in runner.interpreter.get_output_details()]
    arrays = {f"input_{i}": fed[name] for i, name in enumerate(order_in)}
    arrays |= {f"output_{i}": produced[name] for i, name in enumerate(order_out)}
    buffer = io.BytesIO()
    np.savez(buffer, **cast(dict[str, Any], arrays))  # plain savez: zlib versions cannot change the bytes
    return GoldenRecord(
        file=_write(path, buffer.getvalue(), out),
        kind="sequence",
        steps=len(frames),
        resets=tuple(resets),
        source=source,
    )


def _entry(
    name: str,
    spec: ExportSpec | None,
    content: bytes,
    reference: np.ndarray | None,
    out: Path,
    resets: tuple[int, ...] = (),
    source: GoldenSource | None = None,
):
    from .litert import tensor_records

    model = _write(out / name / "model.tflite", content, out)
    inputs, outputs = tensor_records(content)
    stateful = any(r.pair is not None for r in inputs)
    reference_record = None
    if reference is not None and stateful:
        if source is None:
            raise ValueError("A streaming reference needs its source")
        golden = _golden(content, reference, resets, source, out / name / "reference" / "golden.npz", out)
        reference_record = ReferenceRecord(golden=golden)
    elif reference is not None:
        from .runner import LiteRTRunner

        runner = LiteRTRunner(content, reference_kernels=True)
        encoded = runner.encode(reference)
        reference_record = ReferenceRecord(
            inputs=_write(out / name / "reference" / "inputs.npy", _npy(encoded), out),
            outputs=_write(out / name / "reference" / "outputs.npy", _npy(runner.run(encoded)), out),
        )
    return ManifestEntry(
        name=name,
        spec=spec,
        model=model,
        inputs=tuple(TensorEntry.from_record(r) for r in inputs),
        outputs=tuple(TensorEntry.from_record(r) for r in outputs),
        reference=reference_record,
        state_scales_tied=state_scales_tied(inputs, outputs),
    )


def unidentified_install(record: EnvironmentRecord) -> str:
    """Why ``record`` does not identify the helia-edge code, and how to install code that does."""
    return (
        f"helia-edge {record.helia_edge} ({record.helia_edge_source} install) does not identify its code; "
        "install a release, or install from git at a commit: "
        "uv pip install 'helia-edge @ git+https://github.com/AmbiqAI/helia-edge@<commit>'"
    )


def run_recipe(
    recipe_path: Path, out_dir: Path, only: Collection[str] | None = None, require_provenance: bool = False
) -> ExportManifest:
    """Regenerate a recipe's exports into ``out_dir`` and write ``out_dir/manifest.json``.

    The model is built in a fresh Keras session (see ``build_model``).

    Args:
        recipe_path: Recipe file (YAML or JSON). Path sources resolve relative to its directory.
        out_dir: Output directory; each export is written to ``<name>/model.tflite``.
        only: Export names to run; all when None.
        require_provenance: Refuse to run unless the manifest will identify the helia-edge code: a release
            installed from a package index, or an install from a git URL at a commit.

    Returns:
        ExportManifest: The written manifest.
    """
    from .api import export_model

    if require_provenance:
        record = environment_record()
        if not record.identified:
            raise ValueError(unidentified_install(record))
    recipe_path, out_dir = Path(recipe_path).resolve(), Path(out_dir).resolve()
    recipe = load_recipe(recipe_path)
    base = recipe_path.parent
    names = ["import"] if isinstance(recipe.model, TfliteImport) else [e.name for e in recipe.exports]
    if only is not None and set(only) - set(names):
        raise ValueError(f"Unknown export names: {sorted(set(only) - set(names))}; the recipe has {names}")
    selected = [e for e in recipe.exports if only is None or e.name in only]
    other = sorted({e.format for e in selected} - {"litert"})
    if other:
        raise ValueError(f"Recipes export the litert format only; got {other}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reference, resets, source = None, (), None
    if recipe.reference is not None:
        reference = load_array(recipe.reference.source, base, recipe.reference.samples)
        resets = recipe.reference.resets
        file = recipe.reference.source.file
        location = f"path:{file.path}" if isinstance(file, PathSource) else file.url
        key = f"#{recipe.reference.source.key}" if recipe.reference.source.key else ""
        source = GoldenSource(uri=location + key, sha256=file.sha256)

    if isinstance(recipe.model, TfliteImport):
        from .litert import tensor_records

        content = fetch(recipe.model.file, base).read_bytes()
        if reference is not None:
            stateful = any(record.pair is not None for record in tensor_records(content)[0])
            _check_resets(resets, len(reference), stateful, "Reference")
        entries = [_entry("import", None, content, reference, out_dir, resets, source)]
    else:
        model = build_model(recipe.model, base)
        signal = _signal_input(model)
        if reference is not None:
            _check_resets(resets, len(reference), signal is not None, "Reference")
        calibration = None
        if recipe.calibration is not None:
            calibration = load_array(recipe.calibration.source, base, recipe.calibration.samples)
            _check_resets(recipe.calibration.resets, len(calibration), signal is not None, "Calibration")
            if signal is not None:
                from .api import stream_calibration

                calibration = stream_calibration(model, {signal: calibration}, recipe.calibration.resets)
        entries = []
        for entry in selected:
            spec = ExportSpec(**entry.model_dump(exclude={"name"}))
            result = export_model(model, spec, calibration if spec.precision in CALIBRATED else None)
            entries.append(_entry(entry.name, spec, result.content, reference, out_dir, resets, source))

    manifest = ExportManifest(
        schema=MANIFEST_SCHEMA,
        recipe=FileRecord(
            path=Path(os.path.relpath(recipe_path, out_dir)).as_posix(),
            sha256=sha256_file(recipe_path),
            bytes=recipe_path.stat().st_size,
        ),
        environment=EnvironmentEntry.from_record(environment_record()),
        entries=tuple(entries),
    )
    manifest.write(out_dir / "manifest.json")
    return manifest


@dataclass
class VerifyReport:
    """Outcome of ``verify_manifest``. ``status`` is ``ok``, ``drift`` or ``env_mismatch``."""

    status: Literal["ok", "drift", "env_mismatch"]
    differences: list[str] = field(default_factory=list)
    environment_differences: list[str] = field(default_factory=list)


def _environment_differences(recorded: EnvironmentEntry, current: EnvironmentEntry) -> list[str]:
    differences = [
        f"{key}: {getattr(recorded, key)} -> {getattr(current, key)}"
        for key in ("helia_edge", "helia_edge_commit", "python", "platform")
        if getattr(recorded, key) != getattr(current, key)
    ]
    # Manifests written before helia_edge_source was recorded have None
    if recorded.helia_edge_source is not None and recorded.helia_edge_source != current.helia_edge_source:
        differences.append(f"helia_edge_source: {recorded.helia_edge_source} -> {current.helia_edge_source}")
    for package in sorted(set(recorded.packages) | set(current.packages)):
        if recorded.packages.get(package) != current.packages.get(package):
            differences.append(f"{package}: {recorded.packages.get(package)} -> {current.packages.get(package)}")
    return differences


def _files(entry: ManifestEntry) -> list[tuple[str, FileRecord]]:
    files = [("model", entry.model)]
    reference = entry.reference
    if reference is not None and reference.inputs is not None and reference.outputs is not None:
        files += [("reference inputs", reference.inputs), ("reference outputs", reference.outputs)]
    if reference is not None and reference.golden is not None:
        files.append(("golden", reference.golden.file))
    return files


def verify_manifest(manifest_path: Path, allow_env_mismatch: bool = False) -> VerifyReport:
    """Check the recipe and files on disk, then the environment, then regenerate and compare.

    A changed recipe or file is ``drift`` whatever the environment; environment differences are
    still listed. Otherwise a different environment (helia-edge, Python, platform or dependency
    versions, or a recorded install source) is ``env_mismatch`` without regenerating, unless
    ``allow_env_mismatch``. Regenerated entries must match the recorded sha256 and size of every file, the spec and the tensor records.
    """
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = ExportManifest.read(manifest_path)
    differences = []
    recipe_path = (root / manifest.recipe.path).resolve()
    if not recipe_path.is_file():
        differences.append(f"recipe {manifest.recipe.path} is missing")
    elif sha256_file(recipe_path) != manifest.recipe.sha256:
        differences.append(f"recipe {manifest.recipe.path} changed")
    for entry in manifest.entries:
        for label, record in _files(entry):
            path = root / record.path
            if not path.is_file() or sha256_file(path) != record.sha256:
                differences.append(f"{entry.name}: {label} file {record.path} is missing or changed")
    environment = _environment_differences(manifest.environment, EnvironmentEntry.from_record(environment_record()))
    if differences:
        return VerifyReport("drift", differences, environment)
    if environment and not allow_env_mismatch:
        return VerifyReport("env_mismatch", [], environment)

    with tempfile.TemporaryDirectory() as directory:
        try:
            fresh = run_recipe(recipe_path, Path(directory), only=[entry.name for entry in manifest.entries])
        except SourceError as exc:
            return VerifyReport("drift", [f"source: {exc}"], environment)
    regenerated = {entry.name: entry for entry in fresh.entries}
    for entry in manifest.entries:
        again = regenerated.get(entry.name)
        if again is None:
            differences.append(f"{entry.name}: not produced by the recipe")
            continue
        for (label, record), (_, new) in zip(_files(entry), _files(again), strict=False):
            if (record.sha256, record.bytes) != (new.sha256, new.bytes):
                differences.append(f"{entry.name}: {label} sha256 {record.sha256[:12]} -> {new.sha256[:12]}")
        for field_name in ("spec", "inputs", "outputs"):
            if getattr(entry, field_name) != getattr(again, field_name):
                differences.append(f"{entry.name}: recorded {field_name} differs from the regenerated one")
        golden, regenerated_golden = (e.reference.golden if e.reference else None for e in (entry, again))
        if (golden is None) != (regenerated_golden is None) or (
            golden is not None
            and regenerated_golden is not None
            and golden.model_dump(exclude={"file"}) != regenerated_golden.model_dump(exclude={"file"})
        ):
            differences.append(f"{entry.name}: recorded golden differs from the regenerated one")
        if (entry.reference is None) != (again.reference is None):
            differences.append(f"{entry.name}: reference presence changed")
    return VerifyReport("drift" if differences else "ok", differences, environment)
