"""Rewrite a weight-only float16 TFLite model into a native float16 graph."""

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

_FLOAT16 = schema.TensorType.FLOAT16
_FLOAT32 = schema.TensorType.FLOAT32
_DEQUANTIZE = schema.BuiltinOperator.DEQUANTIZE
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def to_native_fp16(model_content: bytes) -> bytes:
    """Return a graph whose inputs, weights, activations and outputs are FLOAT16.

    The TFLite float16 optimization stores weights as FLOAT16 behind
    ``DEQUANTIZE`` operators and computes in FLOAT32. This drops each
    FLOAT16 -> FLOAT32 ``DEQUANTIZE``, rewires its consumers to the FLOAT16
    source, and converts every remaining FLOAT32 tensor and constant buffer to
    FLOAT16. Constants outside the float16 range saturate to +/-65504, as in
    TFLite's float16 optimization. Non-float tensors are unchanged. Signature
    tensor indices are not remapped.

    Args:
        model_content (bytes): Weight-only float16 TFLite flatbuffer.

    Returns:
        bytes: Native float16 TFLite flatbuffer.
    """
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(model_content), 0))
    converted: set[int] = set()
    for subgraph in model.subgraphs:
        remap: dict[int, int] = {}
        kept = []
        for op in subgraph.operators:
            opcode = model.operatorCodes[op.opcodeIndex]
            code = max(opcode.builtinCode, opcode.deprecatedBuiltinCode)
            if code == _DEQUANTIZE and subgraph.tensors[op.inputs[0]].type == _FLOAT16:
                remap[op.outputs[0]] = op.inputs[0]
            else:
                kept.append(op)
        subgraph.operators = kept

        def resolve(index: int) -> int:
            while index in remap:
                index = remap[index]
            return index

        for op in subgraph.operators:
            op.inputs = [resolve(i) for i in op.inputs]
            op.outputs = [resolve(i) for i in op.outputs]
        subgraph.inputs = [resolve(i) for i in subgraph.inputs]
        subgraph.outputs = [resolve(i) for i in subgraph.outputs]

        for tensor in subgraph.tensors:
            if tensor.type != _FLOAT32:
                continue
            tensor.type = _FLOAT16
            buffer = model.buffers[tensor.buffer]
            if tensor.buffer in converted:  # shared constant, already converted
                continue
            converted.add(tensor.buffer)
            if buffer.data is not None and len(buffer.data) > 0:
                values = np.frombuffer(bytes(buffer.data), dtype=np.float32)
                values = np.clip(values, -_FLOAT16_MAX, _FLOAT16_MAX).astype(np.float16)
                buffer.data = values.view(np.uint8)

    builder = flatbuffers.Builder(len(model_content))
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())
