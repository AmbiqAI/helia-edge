"""Export entry point: validate, then convert with the exporter for the active backend."""

import numpy as np
import numpy.typing as npt

from .result import ExportResult
from .spec import CALIBRATED, BackendUnavailable, ExportSpec


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
    if calibration.shape[0] == 0:
        raise ValueError("Calibration data is empty")
    fixed = all(d is None or d == c for c, d in zip(calibration.shape[1:], input_shape[1:], strict=False))
    if calibration.ndim != len(input_shape) or not fixed:
        raise ValueError(f"Calibration shape {calibration.shape} does not match model input {tuple(input_shape)}")
    if not np.isfinite(calibration).all():
        raise ValueError("Calibration data contains NaN or infinity")


def export_model(model, spec: ExportSpec, calibration: npt.NDArray | None = None) -> ExportResult:
    """Export a single-input Keras model.

    Args:
        model: Keras model built on the TensorFlow backend.
        spec: What to export.
        calibration: float32 samples along axis 0, shaped like the model input, for A8W8 and A16W8
            only. They are used one at a time in stored order.

    Returns:
        ExportResult: The exported bytes, their sha256, the I/O tensor records and the environment.

    Raises:
        BackendUnavailable: If the active Keras backend has no exporter for ``spec.format``. A
            process cannot switch Keras backend: rebuild the model from its params and weights in a
            process started with ``KERAS_BACKEND=tensorflow``.
        ValueError: If the format is unknown, the model has more than one input, or the calibration
            data is invalid.
    """
    try:
        import keras
    except ModuleNotFoundError as exc:
        raise ImportError("export_model requires Keras with TensorFlow. Install helia-edge[litert].") from exc

    if spec.format != "litert":
        raise ValueError(f"Unknown export format {spec.format!r}; available: 'litert'")
    backend = keras.backend.backend()
    if backend != "tensorflow":
        raise BackendUnavailable(
            f"LiteRT export needs the TensorFlow backend; the active Keras backend is {backend!r}. "
            "Rebuild the model from its params and weights with KERAS_BACKEND=tensorflow."
        )
    if len(model.inputs) != 1:
        raise ValueError(f"export_model supports single-input models; this model has {len(model.inputs)} inputs")
    check_calibration(spec, calibration, tuple(model.input_shape))

    from .litert import export_litert

    return export_litert(model, spec, calibration)
