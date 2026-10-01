"""Export entry point: validate, then convert with the exporter for the active backend."""

from collections.abc import Collection, Mapping

import numpy as np
import numpy.typing as npt

from .result import ExportResult
from .spec import CALIBRATED, BackendUnavailable, ExportSpec, state_input_name, state_output_name, state_pair


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
