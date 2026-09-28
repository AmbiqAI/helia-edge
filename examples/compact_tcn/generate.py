"""Generate seeded compact TCN models and their FP32/INT8 LiteRT exports."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess

import keras
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from tensorflow.lite.python import schema_py_generated as schema

from helia_edge.converters.litert import ConversionType, LiteRTKerasConverter, QuantizationType
from helia_edge.models.tcn import TcnModel, TcnParams


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(array):
    """Hash dtype, shape and C-order bytes, independent of archive metadata."""
    return sha256(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes(order="C"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


SEED = 20260925
WIDTHS = (8, 16)
INPUT_SHAPE = (240, 14)
NUM_CLASSES = 2
TCN = {
    "input_kernel": None,
    "input_norm": "batch",
    "block_type": "sm",
    "blocks": [
        {"depth": 1, "branch": 1, "filters": 8, "kernel": [1, 3], "dilation": [1, 2**stage], "ex_ratio": 1,
         "se_ratio": 4, "dropout": None, "norm": "batch", "activation": "relu6"}
        for stage in range(4)
    ],
    "output_kernel": [1, 1],
    "include_top": True,
    "use_logits": True,
    "output_activation": None,
    "name": "compact_tcn",
}


def build_model(width, tcn=TCN):
    """Build the seeded preset; ``filters`` in ``tcn`` are replaced by ``width``."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    config = copy.deepcopy(tcn)
    for block in config["blocks"]:
        block["filters"] = width
    params = TcnParams.model_validate(config)
    inputs = keras.Input(shape=INPUT_SHAPE, batch_size=1, name="features")
    model = TcnModel.model_from_params(inputs, params, num_classes=NUM_CLASSES)
    if model.output_shape != (1, 240, 2):
        raise ValueError(f"Unexpected output shape {model.output_shape}")
    return model, params.model_dump(mode="json")


def validate_model(model, width):
    """Require the bounded preset's connected source graph before conversion."""
    fields = {
        "InputLayer": ("batch_shape",),
        "Reshape": ("target_shape",),
        "DepthwiseConv2D": ("kernel_size", "strides", "padding", "dilation_rate", "depth_multiplier", "use_bias", "activation", "data_format"),
        "Conv2D": ("filters", "kernel_size", "strides", "padding", "dilation_rate", "use_bias", "activation", "data_format"),
        "BatchNormalization": ("axis", "momentum", "epsilon"),
        "Activation": ("activation",),
        "GlobalAveragePooling2D": ("keepdims", "data_format"),
        "Multiply": (), "Add": (),
    }

    def node(kind, values, *inputs):
        return kind, json.dumps(values, sort_keys=True), inputs

    def conv(x, filters, padding="same", bias=False):
        return node("Conv2D", [filters, [1, 1], [1, 1], padding, [1, 1], bias, "linear", "channels_last"], x)

    def act(x, activation):
        return node("Activation", [activation], x)

    def bn(x):
        return node("BatchNormalization", [-1, 0.99, 0.001], x)

    x = node("InputLayer", [[1, 240, 14]])
    x = node("Reshape", [[1, 240, 14]], x)
    for stage in range(4):
        skip = x
        x = node("DepthwiseConv2D", [[1, 3], [1, 1], "same", [1, 2**stage], 1, False, "linear", "channels_last"], x)
        x = act(bn(x), "relu6")
        x = act(bn(conv(x, width)), "relu6")
        pooled = node("GlobalAveragePooling2D", [True, "channels_last"], x)
        gate = act(conv(act(conv(pooled, width // 4, "valid", True), "relu6"), width, "valid", True), "hard_sigmoid")
        x = node("Multiply", [], x, gate)
        if stage:
            x = node("Add", [], x, skip)
    expected = node("Reshape", [[240, 2]], conv(x, 2, bias=True))
    seen = set()

    def actual(tensor):
        layer = tensor._keras_history.operation
        kind = type(layer).__name__
        if kind not in fields or layer.compute_dtype != "float32":
            raise ValueError("Source violates compact TCN preset layer/dtype")
        seen.add(layer.name)
        config = layer.get_config()
        inputs = [] if kind == "InputLayer" else layer._inbound_nodes[0].input_tensors
        return node(kind, [config[key] for key in fields[kind]], *(actual(t) for t in inputs))

    if width not in (8, 16) or len(model.outputs) != 1 or actual(model.output) != expected or len(seen) != len(model.layers):
        raise ValueError("Source violates compact TCN preset topology/semantics")


def retain_license(output):
    """Retain source/fixture licensing; dependency versions are not a license audit."""
    source = Path(__file__).resolve().parents[2] / "LICENSE"
    shutil.copyfile(source, output / "LICENSE")
    return {"source_spdx": "BSD-3-Clause", "file": "LICENSE", "sha256": sha256(source.read_bytes()),
            "weights": "synthetic seeded initialization; no third-party trained weights",
            "dependencies": "versions recorded separately; dependency license audit not provided"}


def calibration_inputs(samples):
    """Deterministic uniform [-1, 1] windows used only for INT8 calibration and export checks."""
    if type(samples) is not int or not 1 <= samples <= 256:
        raise ValueError("calibration samples must be in [1, 256]")
    return np.random.default_rng(SEED + 1).uniform(-1.0, 1.0, (samples, *INPUT_SHAPE)).astype(np.float32)


def runtime(content):
    interpreter = Interpreter(model_content=content, num_threads=1,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    return interpreter


def tensor_info(detail):
    quant = detail["quantization_parameters"]
    return {"name": detail["name"], "shape": detail["shape"].tolist(),
            "dtype": np.dtype(detail["dtype"]).name,
            "scales": quant["scales"].tolist(), "zero_points": quant["zero_points"].tolist(),
            "quantized_dimension": int(quant["quantized_dimension"])}


def graph_info(content, precision):
    """Record graph operand types, not unobservable target accumulator/dispatch claims."""
    model = schema.Model.GetRootAsModel(content, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError("Expected one stateless subgraph")
    graph = model.Subgraphs(0)
    type_names = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    op_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    tensors = []
    for i in range(graph.TensorsLength()):
        tensor = graph.Tensors(i)
        quant = tensor.Quantization()
        tensors.append({"index": i, "name": tensor.Name().decode(),
                        "shape": tensor.ShapeAsNumpy().tolist(), "dtype": type_names[tensor.Type()],
                        "constant": model.Buffers(tensor.Buffer()).DataLength() > 0,
                        "quantization": {"scales": quant.ScaleAsNumpy().tolist() if quant.ScaleLength() else [],
                                         "zero_points": quant.ZeroPointAsNumpy().tolist() if quant.ZeroPointLength() else [],
                                         "quantized_dimension": quant.QuantizedDimension()} if quant else None})
    operators = []
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        opcode = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        inputs = [int(x) for x in op.InputsAsNumpy() if x >= 0]
        outputs = [int(x) for x in op.OutputsAsNumpy() if x >= 0]
        spatial = None
        if opcode in (schema.BuiltinOperator.DEPTHWISE_CONV_2D, schema.BuiltinOperator.CONV_2D):
            options = schema.DepthwiseConv2DOptions() if opcode == schema.BuiltinOperator.DEPTHWISE_CONV_2D \
                else schema.Conv2DOptions()
            raw_options = op.BuiltinOptions()
            options.Init(raw_options.Bytes, raw_options.Pos)
            spatial = {"dilation": [options.DilationHFactor(), options.DilationWFactor()],
                       "stride": [options.StrideH(), options.StrideW()],
                       "padding": options.Padding(), "fused_activation": options.FusedActivationFunction()}
        operators.append({"name": op_names[opcode], "version": code.Version(), "spatial_options": spatial,
                          "inputs": inputs, "outputs": outputs,
                          "input_dtypes": [tensors[x]["dtype"] for x in inputs],
                          "output_dtypes": [tensors[x]["dtype"] for x in outputs]})
    depthwise = [op for op in operators if op["name"] == "DEPTHWISE_CONV_2D"]
    if len(depthwise) != 4:
        raise ValueError("Export violates compact TCN preset: expected four depthwise stages")
    for index, op in enumerate(depthwise):
        options = op["spatial_options"]
        kernel = tensors[op["inputs"][1]]["shape"]
        if (options["dilation"] != [1, 2**index] or options["stride"] != [1, 1]
                or options["padding"] != schema.Padding.SAME or kernel[1:3] != [1, 3]):
            raise ValueError("Export violates compact TCN preset kernel/dilation/stride/padding")
    allowed = {"INT8", "INT32"} if precision == "INT8" else {"FLOAT32", "INT32"}
    if precision not in {"INT8", "FP32"} or any(t["dtype"] not in allowed for t in tensors):
        raise ValueError("Unexpected operand dtype (including floating-point in INT8 export)")
    return {"tensors": tensors, "operators": operators,
            "compute_contract": "INT8 activations/weights with integer bias/shape operands" if precision == "INT8"
            else "FP32 data operands with integer shape operands",
            "target_accumulator_and_dispatch": "not measured"}


def infer(interpreter, inputs):
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    result = []
    for sample in inputs:
        interpreter.set_tensor(inp["index"], sample[None])
        interpreter.invoke()
        result.append(interpreter.get_tensor(out["index"])[0])
    return np.stack(result)


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve()]
    files += sorted((root / "helia_edge").rglob("*.py"))
    files += [root / "pyproject.toml", root / "uv.lock"]
    dependencies = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))
    return {"git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "files": {str(p.relative_to(root)): sha256(p.read_bytes()) for p in files},
            "python": platform.python_version(), "dependencies": dependencies,
            "dependencies_sha256": sha256(json.dumps(dependencies, sort_keys=True).encode())}


def generate(output, calibration_samples=32):
    calibration = calibration_inputs(calibration_samples)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "calibration.npy", calibration, allow_pickle=False)
    manifest = {"schema_version": 2, "kind": "seeded synthetic models and exports; no trained task-quality claim",
                "state": "stateless same-padding whole windows; no streaming claim",
                "source": provenance(), "license": retain_license(output),
                "preset": {"seed": SEED, "input_shape": list(INPUT_SHAPE), "num_classes": NUM_CLASSES,
                           "calibration_samples": calibration_samples, "tcn": TCN},
                "calibration_sample_hashes": list(map(array_hash, calibration)), "exports": []}
    for width in WIDTHS:
        model, config = build_model(width)
        validate_model(model, width)
        weights = model.get_weights()
        weight_hashes = list(map(array_hash, weights))
        np.savez(output / f"w{width}-weights.npz", **{f"weight_{i}": w for i, w in enumerate(weights)})
        write_json(output / f"w{width}-config.json", config)
        for precision in ("FP32", "INT8"):
            stem = f"tcn-w{width}-{precision.lower()}"
            converter = LiteRTKerasConverter(model)
            try:
                content = converter.convert(calibration, quantization=QuantizationType(precision),
                                            mode=ConversionType.CONCRETE, strict=True,
                                            io_type="int8" if precision == "INT8" else "float32")
            finally:
                converter.cleanup()
            (output / f"{stem}.tflite").write_bytes(content)
            graph = graph_info(content, precision)
            write_json(output / f"{stem}-graph.json", graph)
            interpreter = runtime(content)
            inp, = interpreter.get_input_details()
            out, = interpreter.get_output_details()
            expected_dtype = np.int8 if precision == "INT8" else np.float32
            if inp["dtype"] != expected_dtype or out["dtype"] != expected_dtype:
                raise ValueError("Export interface dtype mismatch")
            if inp["shape"].tolist() != [1, *INPUT_SHAPE] or out["shape"].tolist() != [1, INPUT_SHAPE[0], NUM_CLASSES]:
                raise ValueError("Export shape mismatch")
            fp32_error = None
            if precision == "FP32":
                exported = infer(interpreter, calibration)
                reference = np.concatenate([model(x[None], training=False).numpy() for x in calibration])
                np.testing.assert_allclose(exported, reference, rtol=1e-5, atol=1e-5)
                fp32_error = float(np.max(np.abs(exported - reference)))
            if weight_hashes != list(map(array_hash, model.get_weights())):
                raise ValueError("Conversion changed the source weights")
            manifest["exports"].append({"width": width, "precision": precision,
                "model": f"{stem}.tflite", "model_sha256": sha256(content), "model_bytes": len(content),
                "graph": f"{stem}-graph.json", "config": f"w{width}-config.json", "weights": f"w{width}-weights.npz",
                "weight_array_hashes": weight_hashes, "parameter_count": model.count_params(),
                "input": tensor_info(inp), "output": tensor_info(out),
                "output_bytes": int(np.prod(out["shape"])) * np.dtype(out["dtype"]).itemsize,
                "fp32_keras_max_abs_error_on_calibration": fp32_error})
    manifest["files"] = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=32)
    args = parser.parse_args()
    manifest = generate(args.output, args.calibration_samples)
    print(json.dumps({"exports": len(manifest["exports"]), "manifest": str(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
