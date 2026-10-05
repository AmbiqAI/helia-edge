"""LiteRT (.tflite) conversion on the TensorFlow backend."""

import contextlib
import hashlib
import tempfile
from collections.abc import Mapping

import flatbuffers
import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

from .fp16 import to_native_fp16
from .result import ExportResult, TensorRecord, environment_record
from .spec import CALIBRATED, ConversionMode, ExportSpec, IODType, Precision, TensorRole, state_pair

_DENSE_PER_TENSOR = "_experimental_disable_per_channel_quantization_for_dense_layers"

_IO_TYPES = {
    schema.TensorType.FLOAT32: IODType.FLOAT32,
    schema.TensorType.FLOAT16: IODType.FLOAT16,
    schema.TensorType.INT8: IODType.INT8,
    schema.TensorType.INT16: IODType.INT16,
}


def convert_litert(
    model: keras.Model,
    *,
    precision: Precision,
    io_dtype: IODType,
    mode: ConversionMode,
    strict: bool,
    calibration: npt.NDArray | Mapping[str, npt.NDArray] | None,
    dense_per_channel: bool,
) -> bytes:
    """Convert a Keras model to LiteRT bytes; ``export_model`` validates the arguments first.

    A model with state inputs is converted with its outputs keyed by output name, so the signature names
    ``state_in_k`` and ``state_out_k``.

    Args:
        model: The Keras model, on the TensorFlow backend.
        precision: The numeric format; FP16 is TensorFlow's float16 weight storage rewritten to a native
            float16 graph (``to_native_fp16``).
        io_dtype: Inputs and outputs of the calibrated precisions. FP32 graphs have float32 and FP16 graphs
            float16 inputs and outputs whatever ``io_dtype`` says, so pass those.
        mode: How the model is traced; SAVED_MODEL exports to a temporary directory removed after conversion.
        strict: For calibrated precisions, refuse operators without an integer kernel.
        calibration: Samples for the calibrated precisions, used one at a time in stored order; for a model
            with several inputs, a mapping of each input name to its samples, fed by name because the
            converter orders inputs its own way.
        dense_per_channel: For calibrated precisions, False quantizes FULLY_CONNECTED weights per tensor
            (convolutions stay per channel).

    Returns:
        bytes: The LiteRT flatbuffer.

    Raises:
        ValueError: If ``mode`` is CONCRETE and the model has several inputs or state inputs: a concrete
            function converts to a graph without a signature, which name-keyed calibration and state
            tensors need. Also if a model with state inputs repeats an output name, as two outputs of one
            layer do.
        ValueError: Also if ``dense_per_channel`` is False and this TensorFlow's converter lacks the
            setting that applies it.
    """
    stateful = any(state_pair(tensor.name) for tensor in model.inputs)
    if mode == ConversionMode.CONCRETE and (stateful or len(model.inputs) > 1):
        raise ValueError(
            "mode 'concrete' converts single-input models only; use 'keras' or 'saved_model' for a model with "
            "several inputs or state inputs"
        )
    if stateful:
        duplicates = sorted({name for name in model.output_names if model.output_names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Output names {duplicates} repeat, as two outputs of one layer do; pass each output of a streaming "
                "model through its own named layer, such as state_output"
            )
        model = keras.Model(model.inputs, dict(zip(model.output_names, model.outputs, strict=True)), name=model.name)

    with contextlib.ExitStack() as stack:  # holds the SavedModel directory until convert()
        match mode:
            case ConversionMode.KERAS:
                converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
            case ConversionMode.SAVED_MODEL:
                workdir = stack.enter_context(tempfile.TemporaryDirectory())
                model.export(workdir, format="tf_saved_model")
                converter = tf.lite.TFLiteConverter.from_saved_model(workdir)
            # Following case is a workaround for bug (https://github.com/tensorflow/tflite-micro/issues/2319)
            # Default TFLiteConverter generates equivalent graph w/ SpaceToBatchND operations but losses dilation_rate factor.
            case ConversionMode.CONCRETE:
                feat_shape = model.input_shape[1:]
                input_shape = (1,) + feat_shape  # Add 1 for batch dimension
                input_spec = tf.TensorSpec(shape=input_shape, dtype=model.input_dtype)
                model_func = tf.function(func=model)
                model_cf = model_func.get_concrete_function(input_spec)
                converter = tf.lite.TFLiteConverter.from_concrete_functions([model_cf])
            case _:
                raise ValueError(f"Invalid conversion mode: {mode}")

        representative_dataset = None
        if isinstance(calibration, Mapping):
            named = dict(calibration)
            steps = len(next(iter(named.values())))

            def representative_dataset():
                """Yield one calibration sample per input name at a time, in stored order."""
                for i in range(steps):
                    yield {name: values[i : i + 1] for name, values in named.items()}

        elif calibration is not None:
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
            case Precision.FP16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            # int8 weights, bias, activation
            case Precision.A8W8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.dtypes.as_dtype(io_dtype.value)
                converter.inference_output_type = tf.dtypes.as_dtype(io_dtype.value)
                converter.representative_dataset = representative_dataset
            # int8 weights, int64 bias, int16 activation
            case Precision.A16W8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
                ]
                converter.inference_input_type = tf.dtypes.as_dtype(io_dtype.value)
                converter.inference_output_type = tf.dtypes.as_dtype(io_dtype.value)
                converter.representative_dataset = representative_dataset

        # Without strict, calibrated precisions fall back to float operators where no integer kernel exists
        if not strict and precision in (Precision.A8W8, Precision.A16W8):
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)
        if not dense_per_channel and precision in (Precision.A8W8, Precision.A16W8):
            # A private converter setting, present in the pinned TensorFlow 2.21; refuse rather than ignore the option without it
            if not hasattr(converter, _DENSE_PER_TENSOR):
                raise ValueError(
                    f"TensorFlow {tf.__version__}'s converter has no {_DENSE_PER_TENSOR}, so dense_per_channel=False "
                    "cannot be applied"
                )
            setattr(converter, _DENSE_PER_TENSOR, True)

        content = converter.convert()
    return to_native_fp16(content) if precision == Precision.FP16 else content


def _signature_names(model) -> tuple[dict[int, str], dict[int, str]]:
    """Tensor index to signature name for the main subgraph's inputs and outputs (empty without one)."""
    for signature in model.signatureDefs or []:
        if signature.subgraphIndex == 0:
            return (
                {int(t.tensorIndex): t.name.decode() for t in signature.inputs or []},
                {int(t.tensorIndex): t.name.decode() for t in signature.outputs or []},
            )
    return {}, {}


def _state_pairs(model, strict: bool = True) -> dict[int, tuple[int, int]]:
    """State pair index to (input tensor, output tensor) from the signature names ``state_in_k``/``state_out_k``.

    With ``strict``, an unpaired state name or a pair whose tensors differ in shape or type is refused;
    otherwise only matching pairs are returned.
    """
    input_names, output_names = _signature_names(model)
    ins = {pair[1]: i for i, name in input_names.items() if (pair := state_pair(name)) and pair[0] == "in"}
    outs = {pair[1]: i for i, name in output_names.items() if (pair := state_pair(name)) and pair[0] == "out"}
    tensors = model.subgraphs[0].tensors

    def matching(k):
        a, b = tensors[ins[k]], tensors[outs[k]]
        return a.type == b.type and list(a.shape) == list(b.shape)

    if strict:
        if ins.keys() != outs.keys():
            raise ValueError(f"State inputs {sorted(ins)} and state outputs {sorted(outs)} do not pair up")
        mismatched = [k for k in sorted(ins) if not matching(k)]
        if mismatched:
            raise ValueError(f"State pairs {mismatched} have inputs and outputs of different shapes or types")
    return {k: (ins[k], outs[k]) for k in sorted(ins.keys() & outs.keys()) if matching(k)}


def tensor_records(content: bytes) -> tuple[tuple[TensorRecord, ...], tuple[TensorRecord, ...]]:
    """Read the main subgraph's input and output tensors from a ``.tflite`` flatbuffer.

    Inputs and outputs are in subgraph order. Tensors whose signature names are ``state_in_k`` and
    ``state_out_k``, with equal shapes and types, are STATE tensors of pair ``k``; the others, including
    unpaired state names of an imported model, are SIGNAL tensors.
    """
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(content), 0))
    subgraph = model.subgraphs[0]
    pairs = {index: k for k, both in _state_pairs(model, strict=False).items() for index in both}

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
            role=TensorRole.STATE if index in pairs else TensorRole.SIGNAL,
            shape=tuple(int(d) for d in (tensor.shapeSignature if tensor.shapeSignature is not None else tensor.shape)),
            dtype=_IO_TYPES[tensor.type],
            scale=float(scales[0]) if len(scales) else None,
            zero_point=int(zero_points[0]) if len(scales) else None,
            pair=pairs.get(index),
        )

    return tuple(record(int(i)) for i in subgraph.inputs), tuple(record(int(i)) for i in subgraph.outputs)


_OPERATOR_NAMES = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}


def _operator_name(model, op) -> str:
    code = model.operatorCodes[op.opcodeIndex]
    return _OPERATOR_NAMES.get(max(code.builtinCode, code.deprecatedBuiltinCode), "CUSTOM")


def operator_names(content: bytes) -> list[str]:
    """Builtin operator names of every subgraph's operators, in execution order."""
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(content), 0))
    return [_operator_name(model, op) for subgraph in model.subgraphs for op in subgraph.operators]


# Kernels that derive their requantization from tensor parameters when prepared, so a state tensor they
# read (or write) may take new parameters. Other kernels, such as RESHAPE, slicing, CONCATENATION and
# pooling, need equal input and output parameters, and LOGISTIC and TANH have a fixed output scale.
# The bias of a reader in _BIASED_READERS is quantized with the input scale, so it is requantized too.
_RESCALING_READERS = frozenset(
    {"ADD", "SUB", "MUL", "FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL", "LOGISTIC", "TANH"}
    | {"RELU", "QUANTIZE", "DEQUANTIZE"}
)
_RESCALING_WRITERS = frozenset(
    {"ADD", "SUB", "MUL", "FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL", "RELU", "QUANTIZE"}
)
_BIASED_READERS = frozenset({"FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D"})
# Kernels that copy their input's bytes, so their output keeps the input's quantization: a tie carries
# through them to their outputs, and on to those outputs' readers.
_PASSTHROUGH = frozenset({"RESHAPE", "SQUEEZE"})


def _tie_group(model, index: int, k: int) -> list[int]:
    """``index`` and the outputs of passthrough readers reached from it, which all take the tied parameters.

    Raises:
        ValueError: If a tensor in the group is read or written by an operator that needs its parameters
            unchanged, or a passthrough output's parameters differ from its input's.
    """
    subgraph = model.subgraphs[0]
    group, pending = [], [index]
    while pending:
        current = pending.pop()
        group.append(current)
        name = subgraph.tensors[current].name.decode()
        writers = {_operator_name(model, op) for op in subgraph.operators if current in list(op.outputs)}
        if current == index and writers - _RESCALING_WRITERS:
            raise ValueError(
                f"Cannot tie state pair {k}: tensor {name!r} is written by {', '.join(sorted(writers - _RESCALING_WRITERS))}, "
                "which needs its scale and zero point unchanged"
            )
        for op in subgraph.operators:
            if current not in list(op.inputs):
                continue
            kind = _operator_name(model, op)
            if kind in _PASSTHROUGH and list(op.inputs)[0] == current:
                output = int(op.outputs[0])
                if _quantization(subgraph.tensors[output]) != _quantization(subgraph.tensors[current]):
                    raise ValueError(f"Cannot tie state pair {k}: {kind} after {name!r} changes its quantization")
                pending.append(output)
            elif kind not in _RESCALING_READERS:
                raise ValueError(
                    f"Cannot tie state pair {k}: tensor {name!r} is used by {kind}, which needs its scale and zero "
                    "point unchanged"
                )
    return group


_INT_RANGES = {schema.TensorType.INT8: np.iinfo(np.int8), schema.TensorType.INT16: np.iinfo(np.int16)}
_BIAS_TYPES = {schema.TensorType.INT32: np.int32, schema.TensorType.INT64: np.int64}


def _requantize_biases(model, index: int, scale: float, k: int) -> None:
    """Requantize the bias of every FULLY_CONNECTED or convolution reading tensor ``index`` to input ``scale``.

    The converter quantizes such a bias with the input scale times the weight scale of each channel, so a
    new input scale needs the bias values rescaled to keep their real values.
    """
    subgraph = model.subgraphs[0]
    name = subgraph.tensors[index].name.decode()
    for op in subgraph.operators:
        inputs = [int(i) for i in op.inputs]
        if index not in inputs or _operator_name(model, op) not in _BIASED_READERS:
            continue
        if inputs[0] != index or index in inputs[1:]:
            raise ValueError(f"Cannot tie state pair {k}: {name!r} is a weight or bias of {_operator_name(model, op)}")
        if len(inputs) < 3 or inputs[2] < 0:
            continue
        bias, weights = subgraph.tensors[inputs[2]], subgraph.tensors[inputs[1]]
        shared = (
            sum(t.buffer == bias.buffer for t in subgraph.tensors) > 1
            or sum(inputs[2] in list(other.inputs) for other in subgraph.operators) > 1
        )
        weight_scale = None if weights.quantization is None else weights.quantization.scale
        if shared or bias.type not in _BIAS_TYPES or weight_scale is None or len(weight_scale) == 0:
            raise ValueError(
                f"Cannot tie state pair {k}: the bias of the operator reading {name!r} cannot be requantized"
            )
        dtype = np.dtype(_BIAS_TYPES[bias.type])
        old_scale = np.asarray(bias.quantization.scale, dtype=np.float64)
        new_scale = (np.float64(scale) * np.asarray(weight_scale, dtype=np.float64)).astype(np.float32)
        buffer = model.buffers[bias.buffer]
        values = np.frombuffer(bytes(bytearray(buffer.data)), dtype=dtype.newbyteorder("<")).astype(np.float64)
        if {old_scale.size, new_scale.size} - {1, values.size}:
            raise ValueError(f"Cannot tie state pair {k}: the bias of the reader of {name!r} has unexpected scales")
        requantized = np.rint(values * old_scale / new_scale.astype(np.float64))
        info = np.iinfo(dtype)
        if requantized.min() < info.min or requantized.max() > info.max:
            raise ValueError(f"Cannot tie state pair {k}: the requantized bias of the reader of {name!r} overflows")
        buffer.data = np.frombuffer(requantized.astype(dtype.newbyteorder("<")).tobytes(), dtype=np.uint8)
        bias.quantization.scale = new_scale


def _quantization(tensor) -> tuple[float, int] | None:
    q = tensor.quantization
    if q is None or q.scale is None or len(q.scale) == 0:
        return None
    return float(q.scale[0]), int(q.zeroPoint[0])


def _covering_quantization(a: tuple[float, int], b: tuple[float, int], info: np.iinfo) -> tuple[float, int]:
    """(scale, zero point) of ``a`` or ``b`` if its range covers the other's, else one covering both ranges."""

    def bounds(q):
        return (info.min - q[1]) * q[0], (info.max - q[1]) * q[0]

    (low_a, high_a), (low_b, high_b) = bounds(a), bounds(b)
    if low_a <= low_b and high_a >= high_b:
        return a
    if low_b <= low_a and high_b >= high_a:
        return b
    low, high = min(low_a, low_b), max(high_a, high_b)
    # One step of margin lets the integer zero point round up and still cover both ends
    exact = (high - low) / (int(info.max) - int(info.min) - 1)
    scale = np.float32(exact)
    if scale < exact:
        scale = np.nextafter(scale, np.float32(np.inf))
    zero = int(np.ceil(info.min - low / float(scale)))
    return float(scale), zero


def tie_state_scales(content: bytes, tolerance: float) -> bytes:
    """Give both tensors of each integer state pair one scale and zero point.

    A runtime carries ``state_out_k`` back into ``state_in_k`` as raw integers, which keeps the state's
    value only if both tensors have the same quantization. For each pair whose parameters differ, both
    tensors take the parameters of the tensor whose range covers the other's (the larger scale of a
    symmetric int16 pair), or else parameters whose range covers both ranges. The kernels reading or
    writing either tensor recompute their scaling from the new parameters when prepared, and the bias
    of a FULLY_CONNECTED or convolution reading a state tensor is requantized to its new input scale. A
    RESHAPE or SQUEEZE reading a state tensor copies its bytes, so its output takes the same parameters,
    and its own readers are checked and requantized in turn.

    Args:
        content: Calibrated ``.tflite`` flatbuffer.
        tolerance: Largest difference between the tied scale and either original scale, relative to that
            original.

    Returns:
        bytes: ``content`` itself when every pair is already tied or float; otherwise the rewritten model.

    Raises:
        ValueError: If state names do not pair up or a pair's tensors differ in shape or type, a pair mixes
            float and integer quantization, the tied scale differs from an original scale by more than
            ``tolerance``, an operator that reads or writes a state tensor needs its parameters unchanged,
            or a state tensor is a reader's weight or bias, or a reader's bias cannot be requantized
            (shared, of an unexpected type or scale count, or overflowing).
    """
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(content), 0))
    subgraph = model.subgraphs[0]
    changed = False
    for k, (index_in, index_out) in _state_pairs(model).items():
        tensor_in, tensor_out = subgraph.tensors[index_in], subgraph.tensors[index_out]
        q_in, q_out = _quantization(tensor_in), _quantization(tensor_out)
        if q_in is None and q_out is None:
            continue
        if q_in is None or q_out is None or tensor_in.type not in _INT_RANGES:
            raise ValueError(f"State pair {k} must have two integer tensors of one type to tie")
        if q_in == q_out:
            continue
        info = _INT_RANGES[tensor_in.type]
        tied = _covering_quantization(q_in, q_out, info)
        difference = max(abs(tied[0] - scale) / scale for scale in (q_in[0], q_out[0]))
        if difference > tolerance:
            raise ValueError(
                f"State pair {k} scales {q_in[0]:.6g} and {q_out[0]:.6g} tie to {tied[0]:.6g}, a difference of "
                f"{difference:.3%}, more than state_tie_tolerance {tolerance:.3%}; calibrate the state inputs with "
                "the states the model produces"
            )
        groups = [(_tie_group(model, index, k), original) for index, original in ((index_in, q_in), (index_out, q_out))]
        for group, original in groups:
            for index in group:
                if original[0] != tied[0]:
                    _requantize_biases(model, index, tied[0], k)
                tensor = subgraph.tensors[index]
                tensor.quantization.scale = np.array([tied[0]], dtype=np.float32)
                tensor.quantization.zeroPoint = np.array([tied[1]], dtype=np.int64)
        changed = True
    if not changed:
        return content
    builder = flatbuffers.Builder(len(content))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def export_litert(
    model: keras.Model, spec: ExportSpec, calibration: npt.NDArray | Mapping[str, npt.NDArray] | None
) -> ExportResult:
    """Export with an already validated spec and calibration array."""
    content = convert_litert(
        model,
        precision=spec.precision,
        io_dtype=spec.io_dtype,
        mode=spec.mode,
        strict=spec.strict,
        calibration=calibration,
        dense_per_channel=spec.dense_per_channel,
    )
    # Refuse unpaired or mismatched state tensors, which tensor_records would record as signals
    _state_pairs(schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(content), 0)))
    if spec.precision in CALIBRATED:
        content = tie_state_scales(content, spec.state_tie_tolerance)
    inputs, outputs = tensor_records(content)
    return ExportResult(
        spec=spec,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        inputs=inputs,
        outputs=outputs,
        environment=environment_record(),
    )
