"""Native float16 conversion: graph invariants against the weight-only float16 export."""

import flatbuffers
import keras
import numpy as np
import pytest
from tensorflow.lite.python import schema_py_generated as schema

from helia_edge.converters.litert import ConversionType, LiteRTKerasConverter, QuantizationType
from helia_edge.converters.tflite import to_native_fp16

OPS = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
TYPES = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}


def build_model():
    keras.utils.set_random_seed(7)
    inputs = keras.Input((16, 16, 3), batch_size=1)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.DepthwiseConv2D(3, padding="same")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(4, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def convert(model, quantization, **kwargs):
    converter = LiteRTKerasConverter(model)
    try:
        return converter.convert(quantization=quantization, mode=ConversionType.CONCRETE, **kwargs)
    finally:
        converter.cleanup()


def graph(content):
    model = schema.Model.GetRootAsModel(content, 0)
    sub = model.Subgraphs(0)
    ops = []
    for i in range(sub.OperatorsLength()):
        code = model.OperatorCodes(sub.Operators(i).OpcodeIndex())
        ops.append(OPS[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())])
    tensors = [sub.Tensors(i) for i in range(sub.TensorsLength())]
    io = [TYPES[tensors[sub.Inputs(0)].Type()], TYPES[tensors[sub.Outputs(0)].Type()]]
    constants = []
    for tensor in tensors:
        data = model.Buffers(tensor.Buffer()).DataAsNumpy()
        if not isinstance(data, int) and data.size:
            constants.append((TYPES[tensor.Type()], data.tobytes()))
    return ops, [TYPES[t.Type()] for t in tensors], io, constants


@pytest.fixture(scope="module")
def exports():
    model = build_model()
    return convert(model, QuantizationType.FP16), convert(model, QuantizationType.FP16_NATIVE)


def test_native_graph_is_float16_end_to_end(exports):
    weight_only, native = exports
    ops, types, io, _ = graph(native)
    assert "DEQUANTIZE" not in ops
    assert "FLOAT32" not in types
    assert io == ["FLOAT16", "FLOAT16"]
    assert graph(weight_only)[2] == ["FLOAT32", "FLOAT32"]


def test_same_operators_as_weight_only_export_without_dequantize(exports):
    weight_only, native = exports
    assert [op for op in graph(weight_only)[0] if op != "DEQUANTIZE"] == graph(native)[0]


def test_constants_are_float16_of_the_weight_only_export(exports):
    weight_only, native = exports

    def as_float16(dtype, raw):
        if dtype == "FLOAT32":
            return np.frombuffer(raw, np.float32).astype(np.float16).tobytes()
        return raw

    expected = sorted(as_float16(dtype, raw) for dtype, raw in graph(weight_only)[3])
    assert sorted(raw for _, raw in graph(native)[3]) == expected


def test_rewrite_is_deterministic_and_idempotent(exports):
    weight_only, native = exports
    assert to_native_fp16(weight_only) == native
    assert graph(to_native_fp16(native))[:3] == graph(native)[:3]


def test_fp16_weight_storage_is_unchanged(exports):
    weight_only, _ = exports
    ops, types, io, _ = graph(weight_only)
    assert "DEQUANTIZE" in ops and "FLOAT16" in types and "FLOAT32" in types


def test_native_rejects_other_io_types():
    with pytest.raises(ValueError, match="float16"):
        convert(build_model(), QuantizationType.FP16_NATIVE, io_type="float32")


def test_shared_constant_buffer_is_converted_once():
    inputs = keras.Input((4,), batch_size=1)
    x = keras.layers.Dense(4, name="a")(inputs)
    outputs = keras.layers.Dense(4, name="b")(x)
    model = keras.Model(inputs, outputs)
    kernel = np.arange(16, dtype=np.float32).reshape(4, 4) / 16 + 0.1
    model.get_layer("a").set_weights([kernel, np.zeros(4, np.float32)])
    model.get_layer("b").set_weights([kernel.T.copy(), np.zeros(4, np.float32)])
    flat = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(convert(model, QuantizationType.FP32)), 0))
    sub = flat.subgraphs[0]
    weights = [
        t
        for t in sub.tensors
        if t.type == schema.TensorType.FLOAT32
        and flat.buffers[t.buffer].data is not None
        and len(flat.buffers[t.buffer].data) == 64
    ]
    assert len(weights) == 2
    weights[1].buffer = weights[0].buffer  # alias both kernels to one buffer
    original = np.frombuffer(bytes(flat.buffers[weights[0].buffer].data), np.float32)
    builder = flatbuffers.Builder(0)
    builder.Finish(flat.Pack(builder), file_identifier=b"TFL3")
    native = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(bytearray(to_native_fp16(bytes(builder.Output()))), 0)
    )
    data = bytes(native.buffers[weights[0].buffer].data)
    assert data == original.astype(np.float16).tobytes()


def test_float32_constants_are_converted_to_float16_values():
    fp32 = convert(build_model(), QuantizationType.FP32)
    expected = sorted(
        np.frombuffer(raw, np.float32).astype(np.float16).tobytes() if dtype == "FLOAT32" else raw
        for dtype, raw in graph(fp32)[3]
    )
    native = to_native_fp16(fp32)
    assert "FLOAT32" not in graph(native)[1]
    assert sorted(raw for _, raw in graph(native)[3]) == expected


def test_predict_rejects_native_float16():
    pytest.importorskip("ai_edge_litert.interpreter")
    converter = LiteRTKerasConverter(build_model())
    try:
        converter.convert(quantization=QuantizationType.FP16_NATIVE, mode=ConversionType.CONCRETE)
        with pytest.raises(ValueError, match="float16 kernels"):
            converter.predict(np.zeros((1, 16, 16, 3), np.float32))
    finally:
        converter.cleanup()
