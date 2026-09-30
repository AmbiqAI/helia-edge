"""LiteRT (.tflite) conversion on the TensorFlow backend."""

import hashlib
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import keras
import numpy.typing as npt
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from ..converters.tflite.fp16 import to_native_fp16
from .result import ExportResult, TensorRecord, environment_record
from .spec import ConversionMode, ExportSpec, IODType, Precision, TensorRole

_IO_TYPES = {
    schema.TensorType.FLOAT32: IODType.FLOAT32,
    schema.TensorType.FLOAT16: IODType.FLOAT16,
    schema.TensorType.INT8: IODType.INT8,
    schema.TensorType.INT16: IODType.INT16,
}


@dataclass(frozen=True)
class Conversion:
    """Converted bytes plus the converter state the legacy quantization debugger reuses."""

    content: bytes
    converter: tf.lite.TFLiteConverter
    representative_dataset: Callable[[], Iterator[list[npt.NDArray]]] | None


def convert_litert(
    model: keras.Model,
    *,
    precision: Precision,
    io_type: str | None,
    mode: ConversionMode,
    strict: bool,
    calibration: npt.NDArray | None,
    workdir: str,
) -> Conversion:
    """Convert a Keras model; shared by ``export_model`` and the legacy converters.

    ``io_type`` applies to calibrated precisions only; when None, A8W8 uses int8 and A16W8 float32
    (the legacy defaults). FP16 graphs always have float16 inputs and outputs. ``workdir`` holds the
    SavedModel for ``ConversionMode.SAVED_MODEL`` and must outlive the returned converter.
    """
    feat_shape = model.input_shape[1:]
    input_shape = (1,) + feat_shape  # Add 1 for batch dimension
    input_spec = tf.TensorSpec(shape=input_shape, dtype=model.input_dtype)

    match mode:
        case ConversionMode.KERAS:
            converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
        case ConversionMode.SAVED_MODEL:
            model.export(workdir, format="tf_saved_model")
            converter = tf.lite.TFLiteConverter.from_saved_model(workdir)
        # Following case is a workaround for bug (https://github.com/tensorflow/tflite-micro/issues/2319)
        # Default TFLiteConverter generates equivalent graph w/ SpaceToBatchND operations but losses dilation_rate factor.
        case ConversionMode.CONCRETE:
            model_func = tf.function(func=model)
            model_cf = model_func.get_concrete_function(input_spec)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([model_cf])
        case _:
            raise ValueError(f"Invalid conversion mode: {mode}")

    representative_dataset = None
    if calibration is not None:
        data = calibration

        def representative_dataset():
            """Yield calibration samples one at a time, in stored order."""
            for i in range(data.shape[0]):
                yield [data[i : i + 1]]

    match precision:
        # float32 weights, bias, activation
        case Precision.FP32:
            pass
        # float16 weights; FP16 is rewritten to float16 activations and IO after conversion
        case Precision.FP32_FP16W | Precision.FP16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        # int8 weights, bias, activation
        case Precision.A8W8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            io_dtype = tf.dtypes.as_dtype(io_type) if io_type else tf.int8
            converter.inference_input_type = io_dtype
            converter.inference_output_type = io_dtype
            converter.representative_dataset = representative_dataset
        # int8 weights, int64 bias, int16 activation
        case Precision.A16W8:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
            ]
            io_dtype = tf.dtypes.as_dtype(io_type) if io_type else tf.float32
            converter.inference_input_type = io_dtype
            converter.inference_output_type = io_dtype
            converter.representative_dataset = representative_dataset

    # Without strict, calibrated precisions fall back to float operators where no integer kernel exists
    if not strict and precision in (Precision.A8W8, Precision.A16W8):
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)

    content = converter.convert()
    if precision == Precision.FP16:
        content = to_native_fp16(content)
    return Conversion(content=content, converter=converter, representative_dataset=representative_dataset)


def tensor_records(content: bytes) -> tuple[tuple[TensorRecord, ...], tuple[TensorRecord, ...]]:
    """Read the main subgraph's input and output tensors from a ``.tflite`` flatbuffer."""
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(content), 0))
    subgraph = model.subgraphs[0]

    def record(index: int) -> TensorRecord:
        tensor = subgraph.tensors[index]
        if tensor.type not in _IO_TYPES:
            raise ValueError(f"Unsupported I/O tensor type {tensor.type} for {tensor.name!r}")
        scales = [] if tensor.quantization is None or tensor.quantization.scale is None else tensor.quantization.scale
        zero_points = (
            []
            if tensor.quantization is None or tensor.quantization.zeroPoint is None
            else tensor.quantization.zeroPoint
        )
        if len(scales) > 1:
            raise ValueError(f"I/O tensor {tensor.name!r} is per-channel quantized; per-tensor is required")
        name = tensor.name.decode() if isinstance(tensor.name, bytes) else str(tensor.name)
        return TensorRecord(
            name=name,
            role=TensorRole.SIGNAL,
            shape=tuple(int(d) for d in tensor.shape),
            dtype=_IO_TYPES[tensor.type],
            scale=float(scales[0]) if len(scales) else None,
            zero_point=int(zero_points[0]) if len(scales) else None,
        )

    return tuple(record(int(i)) for i in subgraph.inputs), tuple(record(int(i)) for i in subgraph.outputs)


def export_litert(model: keras.Model, spec: ExportSpec, calibration: npt.NDArray | None) -> ExportResult:
    """Export with an already validated spec and calibration array."""
    with tempfile.TemporaryDirectory() as workdir:
        content = convert_litert(
            model,
            precision=spec.precision,
            io_type=spec.io_dtype.value,
            mode=spec.mode,
            strict=spec.strict,
            calibration=calibration,
            workdir=workdir,
        ).content
    inputs, outputs = tensor_records(content)
    return ExportResult(
        spec=spec,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        inputs=inputs,
        outputs=outputs,
        environment=environment_record(),
    )
