"""Native float16 export: graph invariants against TensorFlow's weight-only float16 conversion."""

import flatbuffers
import keras
import numpy as np
import pydantic
import pytest

if keras.backend.backend() != "tensorflow":
    pytest.skip("LiteRT export runs on the TensorFlow backend", allow_module_level=True)

import tensorflow as tf  # noqa: E402
from tensorflow.lite.python import schema_py_generated as schema  # noqa: E402

from helia_edge.export import ExportSpec, export_model  # noqa: E402
from helia_edge.export.fp16 import to_native_fp16  # noqa: E402

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


def fp16_weights(model, mode="concrete"):
    """TensorFlow's float16 weight storage with float32 compute: the graph the native rewrite starts from."""
    if mode == "keras":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        spec = tf.TensorSpec((1, *model.input_shape[1:]), model.input_dtype)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([tf.function(model).get_concrete_function(spec)])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def export(model, precision, mode="concrete"):
    io_dtype = {"fp32": "float32", "fp16": "float16"}[precision]
    return export_model(model, ExportSpec(precision=precision, io_dtype=io_dtype, mode=mode)).content


def unpack(content):
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(content), 0))


def indices(values):
    return [] if values is None or isinstance(values, int) else [int(i) for i in values]


def unreferenced_tensors(content):
    """(subgraph, tensor name) of tensors no operator, subgraph input or output references."""
    model = unpack(content)
    unreferenced = []
    for index, sub in enumerate(model.subgraphs):
        used = set(indices(sub.inputs)) | set(indices(sub.outputs))
        for op in sub.operators:
            used |= set(indices(op.inputs)) | set(indices(op.outputs)) | set(indices(op.intermediates))
        unreferenced += [(index, sub.tensors[i].name) for i in range(len(sub.tensors)) if i not in used]
    return unreferenced


def unused_opcodes(content):
    model = unpack(content)
    used = {op.opcodeIndex for sub in model.subgraphs for op in sub.operators}
    return [
        OPS[max(c.builtinCode, c.deprecatedBuiltinCode)] for i, c in enumerate(model.operatorCodes) if i not in used
    ]


def operator_tensor_names(content):
    """Per operator: its op name and the names of its input and output tensors, DEQUANTIZE folded away."""
    model = unpack(content)
    result = []
    for sub in model.subgraphs:
        source = {}
        for op in sub.operators:
            code = model.operatorCodes[op.opcodeIndex]
            if max(code.builtinCode, code.deprecatedBuiltinCode) == schema.BuiltinOperator.DEQUANTIZE:
                source[int(op.outputs[0])] = int(op.inputs[0])

        def name(i):
            while i in source:
                i = source[i]
            return None if i < 0 else sub.tensors[i].name

        for op in sub.operators:
            code = model.operatorCodes[op.opcodeIndex]
            opname = OPS[max(code.builtinCode, code.deprecatedBuiltinCode)]
            if opname != "DEQUANTIZE":
                result.append((opname, [name(i) for i in indices(op.inputs)], [name(i) for i in indices(op.outputs)]))
    return result


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
    return fp16_weights(model), export(model, "fp16")


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


def test_fp16_export_is_the_rewritten_weight_only_conversion(exports):
    weight_only, native = exports
    assert to_native_fp16(weight_only) == native
    assert graph(to_native_fp16(native))[:3] == graph(native)[:3]


def test_fp16_weight_storage_is_unchanged(exports):
    weight_only, _ = exports
    ops, types, io, _ = graph(weight_only)
    assert "DEQUANTIZE" in ops and "FLOAT16" in types and "FLOAT32" in types


def test_native_refuses_other_io_types():
    with pytest.raises(pydantic.ValidationError, match="not valid for"):
        ExportSpec(precision="fp16", io_dtype="float32", mode="concrete")


def test_shared_constant_buffer_is_converted_once():
    inputs = keras.Input((4,), batch_size=1)
    x = keras.layers.Dense(4, name="a")(inputs)
    outputs = keras.layers.Dense(4, name="b")(x)
    model = keras.Model(inputs, outputs)
    kernel = np.arange(16, dtype=np.float32).reshape(4, 4) / 16 + 0.1
    model.get_layer("a").set_weights([kernel, np.zeros(4, np.float32)])
    model.get_layer("b").set_weights([kernel.T.copy(), np.zeros(4, np.float32)])
    flat = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(export(model, "fp32")), 0))
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
    fp32 = export(build_model(), "fp32")
    expected = sorted(
        np.frombuffer(raw, np.float32).astype(np.float16).tobytes() if dtype == "FLOAT32" else raw
        for dtype, raw in graph(fp32)[3]
    )
    native = to_native_fp16(fp32)
    assert "FLOAT32" not in graph(native)[1]
    assert sorted(raw for _, raw in graph(native)[3]) == expected


def test_out_of_range_float32_constants_saturate():
    inputs = keras.Input((2,), batch_size=1)
    outputs = keras.ops.minimum(keras.layers.Dense(2)(inputs), 1e5)
    native = to_native_fp16(export(keras.Model(inputs, outputs), "fp32"))
    values = np.concatenate([np.frombuffer(raw, np.float16) for dtype, raw in graph(native)[3] if dtype == "FLOAT16"])
    assert np.isfinite(values).all() and values.max() == np.finfo(np.float16).max


def test_native_graph_has_no_orphan_tensors_or_unused_opcodes(exports):
    weight_only, native = exports
    assert unreferenced_tensors(weight_only) == []
    assert unreferenced_tensors(native) == []
    assert unused_opcodes(weight_only) == []
    assert unused_opcodes(native) == []


def test_native_operators_read_the_same_tensors_as_weight_only_export(exports):
    weight_only, native = exports
    assert operator_tensor_names(native) == operator_tensor_names(weight_only)


def subgraph_io_names(content):
    model = unpack(content)
    return [
        ([sub.tensors[i].name for i in indices(sub.inputs)], [sub.tensors[i].name for i in indices(sub.outputs)])
        for sub in model.subgraphs
    ]


def test_subgraph_inputs_and_outputs_keep_their_tensors(exports):
    weight_only, native = exports
    assert subgraph_io_names(native) == subgraph_io_names(weight_only)


def test_inputs_placed_after_dropped_tensors_are_renumbered(exports):
    weight_only, _ = exports
    model = unpack(weight_only)
    sub = model.subgraphs[0]
    moved = int(sub.inputs[0])
    order = [i for i in range(len(sub.tensors)) if i != moved] + [moved]  # input tensor becomes the last index
    position = {old: new for new, old in enumerate(order)}
    sub.tensors = [sub.tensors[i] for i in order]
    for op in sub.operators:
        op.inputs = [position[i] if i >= 0 else i for i in indices(op.inputs)]
        op.outputs = [position[i] for i in indices(op.outputs)]
    sub.inputs = [position[i] for i in indices(sub.inputs)]
    sub.outputs = [position[i] for i in indices(sub.outputs)]
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    reordered = bytes(builder.Output())
    native = to_native_fp16(reordered)
    assert subgraph_io_names(native) == subgraph_io_names(reordered)
    assert operator_tensor_names(native) == operator_tensor_names(reordered)


def test_signature_indices_follow_renumbered_tensors():
    model = build_model()
    weight_only = unpack(fp16_weights(model, mode="keras"))
    native = unpack(export(model, "fp16", mode="keras"))
    assert weight_only.signatureDefs and len(native.signatureDefs) == len(weight_only.signatureDefs)
    for before, after in zip(weight_only.signatureDefs, native.signatureDefs, strict=True):
        tensors_before = weight_only.subgraphs[before.subgraphIndex].tensors
        sub_after = native.subgraphs[after.subgraphIndex]
        for io in ("inputs", "outputs"):
            pairs = zip(getattr(before, io), getattr(after, io), strict=True)
            for a, b in pairs:
                assert a.name == b.name
                assert sub_after.tensors[b.tensorIndex].name == tensors_before[a.tensorIndex].name
            assert sorted(t.tensorIndex for t in getattr(after, io)) == sorted(indices(getattr(sub_after, io)))


def test_control_flow_subgraphs_are_pruned_consistently():
    keras.utils.set_random_seed(3)
    inputs = keras.Input((6, 3), batch_size=1)
    outputs = keras.layers.Dense(2)(keras.layers.LSTM(4)(inputs))
    weight_only = fp16_weights(keras.Model(inputs, outputs))
    native = to_native_fp16(weight_only)
    assert len(unpack(native).subgraphs) == len(unpack(weight_only).subgraphs) > 1
    # The converter leaves one unreferenced tensor of its own here; the rewrite must add none.
    assert unreferenced_tensors(native) == unreferenced_tensors(weight_only)
    assert unused_opcodes(native) == []
    assert operator_tensor_names(native) == operator_tensor_names(weight_only)
    assert subgraph_io_names(native) == subgraph_io_names(weight_only)


def as_float32(content):
    """Retype a native float16 graph to float32 so the TFLite interpreter can run it."""
    model = unpack(content)
    done = set()
    for sub in model.subgraphs:
        for tensor in sub.tensors:
            if tensor.type != schema.TensorType.FLOAT16:
                continue
            tensor.type = schema.TensorType.FLOAT32
            buffer = model.buffers[tensor.buffer]
            if tensor.buffer in done or buffer.data is None or not len(buffer.data):
                continue
            done.add(tensor.buffer)
            buffer.data = np.frombuffer(bytes(buffer.data), np.float16).astype(np.float32).view(np.uint8)
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def run(content, x):
    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


@pytest.mark.parametrize("name", ["conv", "lstm"])
def test_pruned_graph_computes_the_weight_only_outputs(name):
    if name == "conv":
        model, shape = build_model(), (1, 16, 16, 3)
    else:
        keras.utils.set_random_seed(3)
        inputs = keras.Input((6, 3), batch_size=1)
        model, shape = keras.Model(inputs, keras.layers.Dense(2)(keras.layers.LSTM(4)(inputs))), (1, 6, 3)
    weight_only = fp16_weights(model)
    native = to_native_fp16(weight_only)
    assert len(unpack(native).subgraphs[0].tensors) < len(unpack(weight_only).subgraphs[0].tensors)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(run(as_float32(native), x), run(weight_only, x), rtol=0, atol=1e-6)
