"""Compare exported topology to exact captured reference graphs, excluding weights/dtypes."""

import copy
import importlib.util
import json
from pathlib import Path

import keras
import numpy as np
import pytest
from tensorflow.lite.python import schema_py_generated as schema

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mlperf_generate", ROOT / "examples/mlperf_tiny/generate.py")
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)
REFERENCES = json.loads((ROOT / "tests/fixtures/mlperf-tiny-reference.json").read_text())


def graph(content):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(content, 0))
    subgraph, = model.subgraphs
    codes = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    producers = {int(i): "input" for i in subgraph.inputs}
    result = []
    for op in subgraph.operators:
        code = model.operatorCodes[op.opcodeIndex]
        name = codes[max(code.builtinCode, code.deprecatedBuiltinCode)]
        inputs = [int(i) for i in op.inputs if i >= 0]
        result.append({"op": name, "inputs": [producers[i] for i in inputs if i in producers],
                       "output_shapes": [subgraph.tensors[i].shape.tolist() for i in op.outputs],
                       "parameter_shapes": [subgraph.tensors[i].shape.tolist() for i in inputs if i not in producers],
                       "options": vars(op.builtinOptions) if op.builtinOptions else None})
        for index in op.outputs:
            producers[int(index)] = len(result) - 1
    return result


def topology(operators):
    # Quantized-bias/quantization flags and opcode versions are precision/runtime-specific.
    fields = {"padding", "strideW", "strideH", "filterWidth", "filterHeight", "dilationWFactor",
              "dilationHFactor", "depthMultiplier", "fusedActivationFunction", "beta", "keepNumDims",
              "weightsFormat"}

    def removable_flatten(index):
        op = operators[index]
        if op["op"] != "RESHAPE" or len(op["inputs"]) != 1 or op["inputs"][0] == "input":
            return False
        incoming = operators[op["inputs"][0]]["output_shapes"]
        outgoing = op["output_shapes"]
        return len(incoming) == len(outgoing) == 1 and incoming[0][:3] == [1, 1, 1] \
            and outgoing[0] == [1, incoming[0][-1]]

    def node(index):
        if index == "input":
            return "input"
        op = operators[index]
        if removable_flatten(index):
            return node(op["inputs"][0])
        parameters = copy.deepcopy(op["parameter_shapes"])
        # LiteRT may omit an optional all-zero Dense bias; source bias values are checked below.
        if op["op"] == "FULLY_CONNECTED" and len(parameters) == 1:
            parameters.append([op["output_shapes"][0][-1]])
        options = {key: value for key, value in (op["options"] or {}).items() if key in fields}
        return (op["op"], op["output_shapes"], parameters, options,
                tuple(node(i) for i in op["inputs"]))

    return sum(not removable_flatten(i) for i in range(len(operators))), node(len(operators) - 1)


@pytest.mark.parametrize("name", generator.BUILDERS)
def test_real_fp32_export_preserves_captured_topology_and_reference_output(name, tmp_path):
    output = tmp_path / name
    manifest = generator.generate(name, output)
    assert manifest["precision"] == "FP32"
    goldens = np.load(output / "goldens.npz")
    expected = goldens["keras_outputs"]
    with pytest.raises(AssertionError):
        generator.check_outputs(np.broadcast_to(expected[0], expected.shape), expected)
    actual = graph((output / "model.tflite").read_bytes())
    source_model = keras.models.load_model(output / "model.keras")
    dense_layers = [layer for layer in source_model.layers if isinstance(layer, keras.layers.Dense)]
    exported_dense = [op for op in actual if op["op"] == "FULLY_CONNECTED"]
    for layer, op in zip(dense_layers, exported_dense, strict=True):
        if len(op["parameter_shapes"]) == 1:
            np.testing.assert_array_equal(layer.bias.numpy(), np.zeros(layer.units, np.float32))
    assert topology(actual) == topology(REFERENCES[name]["capture"]["operators"])
    assert manifest["output"]["shape"] == REFERENCES[name]["capture"]["operators"][-1]["output_shapes"][0]


def test_topology_comparison_detects_wrong_resnet_width_and_disconnected_skip():
    expected = REFERENCES["resnet"]["capture"]["operators"]
    wrong_width = copy.deepcopy(expected)
    wrong_width[0]["parameter_shapes"][0][0] = 16
    assert topology(wrong_width) != topology(expected)
    wrong_skip = copy.deepcopy(expected)
    wrong_skip[3]["inputs"][0] = wrong_skip[3]["inputs"][1]
    assert topology(wrong_skip) != topology(expected)
