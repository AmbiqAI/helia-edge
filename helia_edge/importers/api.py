"""Import weights from a pinned source file into a Keras model through a checked mapping."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mapping import WeightMapping


@dataclass(frozen=True)
class ImportReport:
    """What an import assigned: each model weight's path with the source tensors it came from."""

    mapping: str
    source_sha256: str
    assignments: tuple[tuple[str, tuple[str, ...]], ...]
    unused: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight(model, layer: str, weight: str):
    """The weight ``weight`` (a name, or a path such as ``query/kernel`` inside a composite layer) of ``layer``."""
    target = model
    for name in layer.split("/"):
        if not hasattr(target, "get_layer"):
            raise ValueError(f"Layer path {layer!r}: {target.name!r} has no sublayers; name its weight as a path")
        target = target.get_layer(name)
    by_path = [w for w in target.weights if w.path.endswith("/" + weight)] if "/" in weight else []
    matches = by_path or [w for w in target.weights if w.name == weight]
    if len(matches) != 1:
        paths = [w.path for w in target.weights]
        raise ValueError(f"Layer {layer!r} has {len(matches)} weights matching {weight!r}; its weights are {paths}")
    return matches[0]


def _as_stored(value: np.ndarray, dtype) -> np.ndarray:
    """``value`` as a weight of ``dtype`` stores it, read back as float32 (catches overflow to inf in bfloat16)."""
    import keras

    return keras.ops.convert_to_numpy(keras.ops.cast(keras.ops.convert_to_tensor(value), dtype)).astype(np.float32)


def _numpy_dtype(dtype) -> np.dtype:
    """NumPy dtype to check and assign a Keras weight's value in; float32 for a float type NumPy does not
    treat as float, such as bfloat16 (the variable casts on assignment)."""
    try:
        numpy_dtype = np.dtype(dtype)
    except TypeError:
        numpy_dtype = None
    if numpy_dtype is not None and numpy_dtype.kind in "fiub":
        return numpy_dtype
    if "float" in str(dtype):
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported weight dtype {dtype}")


def import_weights(model, mapping: WeightMapping, path: Path | str) -> ImportReport:
    """Set every weight of ``model`` from the file at ``path`` as ``mapping`` describes.

    Nothing is assigned unless every check passes:

    - the file's sha256 is the mapping's pinned ``source.sha256``;
    - every source tensor is used exactly once (each part once when split), or listed as ``unused``;
    - every weight of ``model`` is assigned exactly once, with its shape, from a source of a matching
      kind (float to float), finite as stored in the weight's dtype (integers within its range).

    Args:
        model: A built Keras model.
        mapping: The mapping for the model's architecture.
        path: Local path to the source file; it is never downloaded.

    Returns:
        ImportReport: The assignments and the source sha256.

    Raises:
        ValueError: If the file does not match its pin or changes while it is read; otherwise if any
            other check fails, with every problem found listed in the message.
    """
    from ..registry import importers

    path = Path(path)
    sha256 = _file_sha256(path)
    if sha256 != mapping.source.sha256:
        raise ValueError(
            f"{path} has sha256 {sha256}; mapping {mapping.name!r} is for {mapping.source.sha256} ({mapping.source.uri})"
        )
    tensors = importers.get(mapping.format)(path)
    if _file_sha256(path) != sha256:
        raise ValueError(f"{path} changed while it was read")

    problems = []
    unknown = sorted({name for row in mapping.rows for name in row.sources} - set(tensors))
    unknown += sorted(set(mapping.unused) - set(tensors))
    if unknown:
        problems.append(f"source tensors not in the file: {unknown}")

    uses: dict[str, list] = {}
    for row in mapping.rows:
        for name in row.sources:
            uses.setdefault(name, []).append(row.split)
    for name, splits in sorted(uses.items()):
        if name in mapping.unused:
            problems.append(f"{name!r} is both mapped and listed as unused")
        elif len(splits) > 1 or splits[0] is not None:
            whole = any(split is None for split in splits)
            layouts = {(split.axis, split.parts) for split in splits if split is not None}
            indexes = sorted(split.index for split in splits if split is not None)
            if whole or len(layouts) != 1 or indexes != list(range(next(iter(layouts))[1])):
                problems.append(f"{name!r} is used more than once, or its split parts are not each used once")
    unused = sorted(set(tensors) - set(uses) - set(mapping.unused))
    if unused:
        problems.append(f"source tensors neither mapped nor listed as unused: {unused}")

    assigned = {}
    reported = set()  # weights whose row already reported a problem
    values = {}
    for row in mapping.rows:
        try:
            weight = _weight(model, row.layer, row.weight)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        reported.add(id(weight))
        if any(name not in tensors for name in row.sources):
            continue
        if id(weight) in assigned:
            problems.append(f"{row.layer}/{row.weight} is the same weight as another row's")
            continue
        try:
            dtype = _numpy_dtype(weight.dtype)
        except ValueError as exc:
            problems.append(f"{row.layer}/{row.weight}: {exc}")
            continue
        kinds = {np.asarray(tensors[name]).dtype.kind for name in row.sources}
        if (kinds != {"f"}) if dtype.kind == "f" else ("f" in kinds):
            problems.append(f"{row.layer}/{row.weight}: source kinds {sorted(kinds)} for a {weight.dtype} weight")
            continue
        try:
            value = row.value(tensors)
        except (ValueError, IndexError) as exc:  # numpy reports a bad axis as an IndexError
            problems.append(f"{row.layer}/{row.weight}: {exc}")
            continue
        if tuple(value.shape) != tuple(weight.shape):
            problems.append(
                f"{row.layer}/{row.weight}: mapped shape {value.shape} for weight shape {tuple(weight.shape)}"
            )
            continue
        if dtype.kind in "iu":
            info = np.iinfo(dtype)
            if value.size and (value.min() < info.min or value.max() > info.max):
                problems.append(f"{row.layer}/{row.weight}: the mapped value is out of range for {weight.dtype}")
                continue
        with np.errstate(over="ignore"):  # an overflow becomes inf, refused below
            value = value.astype(dtype)
        if dtype.kind == "f" and not np.isfinite(_as_stored(value, weight.dtype)).all():
            problems.append(f"{row.layer}/{row.weight}: the mapped value is not finite as {weight.dtype}")
            continue
        reported.discard(id(weight))
        assigned[id(weight)] = (row, weight)
        values[id(weight)] = value

    missing = [w.path for w in model.weights if id(w) not in assigned and id(w) not in reported]
    if missing:
        problems.append(f"model weights without a mapping: {missing}")
    if problems:
        raise ValueError(f"Mapping {mapping.name!r} cannot import {path.name}:\n- " + "\n- ".join(problems))

    for key, (_, weight) in assigned.items():
        weight.assign(values[key])
    return ImportReport(
        mapping=mapping.name,
        source_sha256=sha256,
        assignments=tuple((weight.path, row.sources) for row, weight in assigned.values()),
        unused=tuple(mapping.unused),
    )
