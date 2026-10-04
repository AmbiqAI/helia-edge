"""Export entry points: ``export`` (an artifact with its export record) and ``export_model`` (the
conversion), validating, then converting with the exporter for the active backend."""

import collections
import hashlib
import io
import operator
from collections.abc import Collection, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .record import (
    CalibrationRecord,
    EnvironmentEntry,
    ExportOptions,
    ExportRecord,
    ExportSettings,
    GoldenRecord,
    IORecord,
    Source,
    TensorEntry,
    WeightImport,
    WeightsRecord,
    file_record,
    weights_digest,
)
from .result import ExportResult, state_scales_tied
from .spec import (
    CALIBRATED,
    BackendUnavailable,
    ExportSpec,
    IODType,
    Precision,
    TensorRole,
    check_resets,
    state_input_name,
    state_output_name,
    state_pair,
)


def check_calibration(spec: ExportSpec, calibration: npt.NDArray | None, input_shape: tuple) -> None:
    """Reject missing, unexpected or malformed calibration data.

    Args:
        spec: Export specification.
        calibration: Calibration samples along axis 0, or None.
        input_shape: Model input shape including the batch dimension; None dimensions accept any size.

    Raises:
        ValueError: If calibration is missing for a calibrated precision, given for another
            precision, empty, not float32 (whatever the model input dtype), not finite, or shaped
            differently from the model input's fixed dimensions.
    """
    if spec.precision not in CALIBRATED:
        if calibration is not None:
            raise ValueError(f"{spec.precision.value!r} is not calibrated; pass calibration=None")
        return
    if calibration is None:
        raise ValueError(f"{spec.precision.value!r} requires calibration data")
    if not isinstance(calibration, np.ndarray) or calibration.dtype != np.float32:
        raise ValueError("Calibration data must be a float32 numpy array")
    if calibration.ndim != len(input_shape):
        raise ValueError(f"Calibration shape {calibration.shape} does not match model input {tuple(input_shape)}")
    if calibration.shape[0] == 0:
        raise ValueError("Calibration data is empty")
    fixed = all(d is None or d == c for c, d in zip(calibration.shape[1:], input_shape[1:], strict=True))
    if not fixed:
        raise ValueError(f"Calibration shape {calibration.shape} does not match model input {tuple(input_shape)}")
    if not np.isfinite(calibration).all():
        raise ValueError("Calibration data contains NaN or infinity")


def check_named_calibration(
    spec: ExportSpec, calibration: npt.NDArray | Mapping[str, npt.NDArray] | None, input_shapes: Mapping[str, tuple]
) -> None:
    """Reject calibration for a model with several inputs unless it names every input once.

    Args:
        spec: Export specification.
        calibration: Input name to samples along axis 0, or None; an array is refused.
        input_shapes: Input name to shape including the batch dimension.

    Raises:
        ValueError: As ``check_calibration`` for each input, or if the names differ from the model's
            inputs or the inputs have different numbers of samples.
    """
    if spec.precision not in CALIBRATED:
        if calibration is not None:
            raise ValueError(f"{spec.precision.value!r} is not calibrated; pass calibration=None")
        return
    if calibration is None:
        raise ValueError(f"{spec.precision.value!r} requires calibration data")
    if not isinstance(calibration, Mapping):
        raise ValueError(
            f"A model with inputs {sorted(input_shapes)} needs calibration as a mapping of input name to samples"
        )
    if set(calibration) != set(input_shapes):
        raise ValueError(
            f"Calibration names {sorted(calibration)} do not match the model inputs {sorted(input_shapes)}"
        )
    for name, shape in input_shapes.items():
        check_calibration(spec, calibration[name], shape)
    if len({len(values) for values in calibration.values()}) != 1:
        raise ValueError("Calibration inputs have different numbers of samples")


def stream_calibration(
    model, signals: Mapping[str, npt.NDArray], resets: Collection[int] = ()
) -> dict[str, npt.NDArray]:
    """Calibration for a streaming model: its signal inputs with the states the model produces from them.

    The model is called once per step with batch size 1. Each ``state_in_k`` starts at zero, resets to zero
    at the steps in ``resets``, and otherwise takes the model's ``state_out_k`` of the previous step, so the
    state inputs are calibrated with the values they take when the model streams.

    Args:
        model: Keras model whose state inputs and outputs are named ``state_in_k`` and ``state_out_k``.
        signals: Name to float32 samples along axis 0 (the steps) for every input that is not a state.
        resets: Steps at which the states are zero.

    Returns:
        dict[str, npt.NDArray]: Every input name to its float32 samples, as ``export_model`` takes them.

    Raises:
        ValueError: If the model has no input that is not a state, a state input has no fixed shape or no
            matching output, ``signals`` does not name exactly the inputs that are not states, or the
            signals have different numbers of steps.
    """
    import keras

    names = [tensor.name for tensor in model.inputs]
    pairs = [state_pair(name) for name in names]
    states = {name: pair[1] for name, pair in zip(names, pairs, strict=True) if pair and pair[0] == "in"}
    expected = sorted(set(names) - set(states))
    if not expected:
        raise ValueError("stream_calibration needs a model with at least one input that is not a state")
    unknown = [name for name in states if None in model.inputs[names.index(name)].shape[1:]]
    if unknown:
        raise ValueError(f"State inputs {unknown} need fixed shapes")
    if sorted(signals) != expected:
        raise ValueError(f"Signals {sorted(signals)} do not match the inputs that are not states {expected}")
    missing = sorted(state_output_name(k) for k in states.values() if state_output_name(k) not in model.output_names)
    if missing:
        raise ValueError(f"The model has no outputs {missing} for its state inputs")
    steps = {len(values) for values in signals.values()}
    if len(steps) != 1:
        raise ValueError("Signals have different numbers of steps")
    zeros = {name: np.zeros((1, *model.inputs[names.index(name)].shape[1:]), np.float32) for name in states}
    current = dict(zeros)
    resets = set(resets)
    collected: dict[str, list] = {name: [] for name in names}
    for step in range(steps.pop()):
        if step in resets:
            current = dict(zeros)
        feed = {name: np.asarray(signals[name][step : step + 1], np.float32) for name in signals} | current
        for name in names:
            collected[name].append(feed[name])
        outputs = model([feed[name] for name in names], training=False)
        if not isinstance(outputs, Mapping):
            outputs = dict(
                zip(model.output_names, outputs if isinstance(outputs, list | tuple) else [outputs], strict=True)
            )
        current = {
            state_input_name(k): keras.ops.convert_to_numpy(outputs[state_output_name(k)]).astype(np.float32)
            for k in states.values()
        }
    return {name: np.concatenate(values) for name, values in collected.items()}


def export_model(
    model, spec: ExportSpec, calibration: npt.NDArray | Mapping[str, npt.NDArray] | None = None
) -> ExportResult:
    """Export a Keras model.

    A streaming model names its state inputs and outputs ``state_in_k`` and ``state_out_k``
    (``helia_edge.layers.state_input`` and ``state_output``); they are recorded with role STATE and pair
    ``k``. For calibrated precisions with integer I/O, each pair gets one scale and zero point (see
    ``ExportSpec.state_tie_tolerance``), so a runtime can carry the raw state.

    Args:
        model: Keras model built on a backend with an exporter for ``spec.format``.
        spec: What to export.
        calibration: For A8W8 and A16W8 only: float32 samples along axis 0, shaped like the model input,
            or for a model with several inputs a mapping of every input name to its samples
            (``stream_calibration`` builds one for a streaming model). Samples are used one at a time in
            stored order.

    Returns:
        ExportResult: The exported bytes, their sha256, the I/O tensor records and the environment.

    Raises:
        BackendUnavailable: If ``helia_edge.registry.exporters`` (built-ins and plugins) has no
            exporter for ``spec.format`` on the active Keras backend. A process cannot switch Keras
            backend: rebuild the model from its params and weights in a process started with a
            backend that has one (``KERAS_BACKEND=tensorflow`` for the built-in LiteRT exporter).
        ValueError: If the format is unknown, the calibration data is invalid, or a state pair cannot be
            tied.
        PluginError: If a ``helia_edge.plugins`` entry point fails while plugins load.
    """
    from ..registry import exporters, load_plugins

    def backends_for(fmt: str) -> list[str]:
        return [key.split(":", 1)[1] for key in exporters if key.split(":", 1)[0] == fmt]

    if not backends_for(spec.format):
        load_plugins()
    backends = backends_for(spec.format)
    if not backends:
        formats = sorted({key.split(":", 1)[0] for key in exporters})
        raise ValueError(f"Unknown export format {spec.format!r}; available: {', '.join(map(repr, formats))}")
    try:
        import keras
    except ModuleNotFoundError as exc:
        raise ImportError("export_model requires Keras and a backend. Install helia-edge[litert].") from exc

    backend = keras.backend.backend()
    if backend not in backends:
        load_plugins()
        backends = backends_for(spec.format)
    if backend not in backends:
        raise BackendUnavailable(
            f"{spec.format} export needs the {' or '.join(backends)} backend; the active Keras backend is "
            f"{backend!r}. Rebuild the model from its params and weights with KERAS_BACKEND={backends[0]}."
        )
    if len(model.inputs) == 1:
        if isinstance(calibration, Mapping):
            raise ValueError("A single-input model takes calibration as an array")
        check_calibration(spec, calibration, tuple(model.input_shape))
    else:
        shapes = {tensor.name: tuple(tensor.shape) for tensor in model.inputs}
        check_named_calibration(spec, calibration, shapes)
    return exporters.get(f"{spec.format}:{backend}")(model, spec, calibration)


@contextmanager
def _reference_build():
    """Build with unnamed layers numbered from zero and a float32 dtype policy and ``floatx``, then restore those.

    Unnamed layers' names become the artifact's tensor names, so a model built here exports the same bytes
    whatever the caller built before or set as its policy. These three settings are restored even when the
    build fails; the build itself, like any Keras build, may draw from the random generators. Layer numbering and the dtype policy are per thread, but
    ``floatx`` is shared by the process: another thread building layers meanwhile sees float32.
    """
    import keras
    from keras.src.backend.common import global_state

    names = global_state.get_global_attribute("object_name_uids")
    policy, floatx = keras.config.dtype_policy(), keras.config.floatx()
    global_state.set_global_attribute("object_name_uids", collections.defaultdict(int))
    keras.config.set_floatx("float32")
    keras.config.set_dtype_policy("float32")
    try:
        yield
    finally:
        global_state.set_global_attribute(
            "object_name_uids", names if names is not None else collections.defaultdict(int)
        )
        keras.config.set_floatx(floatx)
        keras.config.set_dtype_policy(policy)


def _steps(resets: Collection[int]) -> tuple[int, ...]:
    """``resets`` as integer steps; a bool, float, string, bytes or non-collection is refused, not converted."""
    if isinstance(resets, str | bytes | set | frozenset):
        raise ValueError(f"resets are integer steps, not {resets!r:.80}")
    try:
        steps = [step for step in resets]
        if any(isinstance(step, bool | np.bool_) for step in steps):
            raise TypeError("bool")
        return tuple(operator.index(step) for step in steps)
    except TypeError as exc:
        raise ValueError(f"resets are integer steps, not {resets!r:.80}") from exc


def _array_sha256(array: npt.ArrayLike) -> str:
    """sha256 of ``array`` as ``numpy.save`` stores it."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@dataclass(frozen=True)
class Export:
    """An exported artifact with its export record.

    Attributes:
        content: The artifact (a ``.tflite`` flatbuffer).
        record: Its export record.
        model: The Keras model that was exported, with the record's batch size.
        golden: The golden@2 NPZ, when ``with_golden`` made one.
    """

    content: bytes
    record: ExportRecord
    model: Any
    golden: bytes | None = None

    def with_golden(self, inputs: npt.NDArray, resets: Collection[int] = (), uri: str | None = None) -> "Export":
        """This export with a golden@2 sequence: ``inputs`` are consecutive calls of the streaming model's
        signal input, run with LiteRT's reference kernels, the state carried and reset at ``resets``.

        Args:
            inputs: The signal, one call per row.
            resets: Calls at which the states are zero.
            uri: Where the inputs can be fetched, if anywhere; recorded with their sha256.
        """
        from .golden import golden_npz

        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.ndim < 1 or len(inputs) == 0 or not np.isfinite(inputs).all():
            raise ValueError("A golden needs at least one call of finite inputs")
        stateful = any(entry.role is TensorRole.STATE for entry in self.record.io.inputs)
        resets = _steps(resets)
        check_resets(resets, len(inputs), stateful, "Golden")
        data = golden_npz(self.content, inputs, resets)
        golden = GoldenRecord(
            file=file_record("golden.npz", data),
            steps=len(inputs),
            resets=resets,
            inputs=Source(sha256=_array_sha256(inputs), uri=uri),
        )
        return replace(self, record=self.record.model_copy(update={"golden": golden}), golden=data)

    def write(self, directory: Path | str) -> Path:
        """Write ``model.tflite``, ``model.weights.h5``, ``golden.npz`` (if any) and ``record.json``, replacing
        those files, and removing a ``golden.npz`` the record does not name.

        Returns:
            Path: The record's path.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / self.record.artifact.file).write_bytes(self.content)
        self.model.save_weights(directory / "model.weights.h5")
        (directory / "golden.npz").unlink(missing_ok=True)  # never leave a golden the record does not name
        if self.golden is not None and self.record.golden is not None:
            (directory / self.record.golden.file.file).write_bytes(self.golden)
        self.record.write(directory / "record.json")
        return directory / "record.json"


def export(
    model,
    *,
    precision: Precision | str,
    io_dtype: IODType | str,
    calibration: npt.NDArray | None = None,
    resets: Collection[int] = (),
    spec=None,
    batch_size: int = 1,
    options: ExportOptions = ExportOptions(),
    weights_import: WeightImport | None = None,
    calibration_uri: str | None = None,
) -> Export:
    """Export a Keras model to LiteRT with its export record (``helia-edge/export-record@1``).

    The exported model has a static batch: every input's batch is ``batch_size``. With ``spec``, the export
    is ``build(spec, batch_size=batch_size)`` with the model's weights, built with unnamed layers numbered
    from zero and a float32 dtype policy and ``floatx`` (then restored), so the artifact does not depend
    on the caller's layer numbering or those settings, and a record re-exports the same bytes. Without
    ``spec``, the model is exported as it is, its batch must be ``batch_size``, and the record cannot
    rebuild it. A streaming model (``state_in_k``/``state_out_k``) is calibrated with
    ``stream_calibration`` from its signal, the states reset at ``resets``.

    Args:
        model: The Keras model.
        precision: ``fp32``, ``fp16``, ``a8w8`` or ``a16w8``.
        io_dtype: Input and output element type, valid for ``precision``.
        calibration: For ``a8w8`` and ``a16w8``: samples along axis 0 of the model's one input, or of a
            streaming model's signal input, cast to float32; the record hashes the float32 array.
        resets: For a streaming model, calibration steps at which the states are zero.
        spec: The model's ``ModelSpec``, recorded so the export can be rebuilt; the model's weights must
            have the shapes of ``build(spec)``.
        batch_size: The batch of the exported model.
        options: LiteRT options.
        weights_import: Where the weights came from, when imported from another framework.
        calibration_uri: Where the calibration array can be fetched, if anywhere.

    Returns:
        Export: The artifact, its record and the exported Keras model.

    Raises:
        ValueError: If the batch differs from ``batch_size`` without ``spec``, the model's weight or input
            shapes differ from ``build(spec)``'s, a calibrated precision has ``batch_size`` other than 1, or
            the calibration or resets are invalid.
    """
    calibration = None if calibration is None else np.asarray(calibration, dtype=np.float32)
    resets = _steps(resets)
    if resets and calibration is None:
        raise ValueError("resets apply to the calibration of a streaming model; no calibration was given")
    settings = ExportSettings(
        precision=Precision(precision),
        io_dtype=IODType(io_dtype),
        batch_size=batch_size,
        options=options,
        calibration=None
        if calibration is None
        else CalibrationRecord(
            sha256=_array_sha256(calibration), uri=calibration_uri, samples=len(calibration), resets=resets
        ),
    )
    if spec is not None:
        from ..models.spec import build

        weights = model.get_weights()
        weight_shapes = [tuple(w.shape) for w in model.weights]
        input_shapes = [tuple(t.shape[1:]) for t in model.inputs]
        with _reference_build():
            rebuilt = build(spec, batch_size=batch_size)
        want = [tuple(w.shape) for w in rebuilt.weights]
        if want != weight_shapes:
            raise ValueError(f"The model's weights do not have the shapes of build(spec): {weight_shapes} vs {want}")
        want = [tuple(t.shape[1:]) for t in rebuilt.inputs]
        if want != input_shapes:
            raise ValueError(f"The model's inputs do not have the shapes of build(spec): {input_shapes} vs {want}")
        rebuilt.set_weights(weights)
        model = rebuilt
    elif {tensor.shape[0] for tensor in model.inputs} != {batch_size}:
        batches = sorted({tensor.shape[0] for tensor in model.inputs}, key=str)
        raise ValueError(
            f"The model's input batch is {batches}, not {batch_size}. Pass spec to export build(spec, "
            f"batch_size={batch_size}) with the model's weights, or build the model with that batch."
        )

    names = [tensor.name for tensor in model.inputs]
    signals = [name for name in names if state_pair(name) is None]
    streaming = len(signals) < len(names)
    if calibration is not None:
        check_resets(resets, len(calibration), streaming, "Calibration")
    data: npt.NDArray | Mapping[str, npt.NDArray] | None = calibration
    if calibration is not None and streaming:
        if len(signals) != 1:
            raise ValueError(f"A streaming model needs exactly one input that is not a state; got {signals}")
        data = stream_calibration(model, {signals[0]: calibration.astype(np.float32)}, resets)
    result = export_model(
        model,
        ExportSpec(
            precision=settings.precision,
            io_dtype=settings.io_dtype,
            mode=options.mode,
            strict=options.strict,
            state_tie_tolerance=options.state_tie_tolerance,
        ),
        data,
    )
    record = ExportRecord(
        model=spec,
        weights=WeightsRecord(digest=weights_digest(model), import_=weights_import),
        export=settings,
        artifact=file_record("model.tflite", result.content),
        io=IORecord(
            inputs=tuple(TensorEntry.from_record(r) for r in result.inputs),
            outputs=tuple(TensorEntry.from_record(r) for r in result.outputs),
            state_scales_tied=state_scales_tied(result.inputs, result.outputs),
        ),
        environment=EnvironmentEntry.from_record(result.environment),
    )
    return Export(content=result.content, record=record, model=model)


def load_export_record(path: Path | str, weights: Path | str | None = None):
    """Rebuild the Keras model an export record describes, with its weights.

    The model is built as ``export`` built it: unnamed layers numbered from zero and float32 layers,
    whatever the caller's dtype policy.

    Args:
        path: The ``record.json``.
        weights: The ``.weights.h5`` file; by default ``model.weights.h5`` next to the record.

    Returns:
        keras.Model: ``build(record.model)`` with the record's batch size and the weights.

    Raises:
        ValueError: If the record has no model spec, or the weights do not have the record's digest.
    """
    from ..models.spec import build

    record = ExportRecord.read(path)
    if record.model is None:
        raise ValueError(f"{path} has no model spec, so the model cannot be rebuilt")
    with _reference_build():  # as export() built it
        model = build(record.model, batch_size=record.export.batch_size)
    model.load_weights(Path(weights) if weights is not None else Path(path).with_name("model.weights.h5"))
    digest = weights_digest(model)
    if digest != record.weights.digest:
        raise ValueError(f"The weights have digest {digest}; the record names {record.weights.digest}")
    return model
