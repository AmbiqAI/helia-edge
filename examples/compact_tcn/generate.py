"""Generate retained, synthetic whole-window TCN performance fixtures."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import keras
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from tensorflow.lite.python import schema_py_generated as schema

from helia_edge.converters.litert import ConversionType, LiteRTKerasConverter, QuantizationType
from helia_edge.models.tcn import TcnBlockParams, TcnModel, TcnParams


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(array):
    """Hash dtype, shape and C-order bytes, independent of archive metadata."""
    return sha256(str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes(order="C"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_recipe(path):
    recipe = json.loads(path.read_text())
    expected = {"schema_version", "seed", "widths", "input_shape", "num_classes", "calibration_samples", "tcn"}
    if set(recipe) != expected or recipe["schema_version"] != 1:
        raise ValueError("Unsupported recipe schema")
    if recipe["input_shape"] != [240, 14] or recipe["num_classes"] != 2:
        raise ValueError("This fixture requires 240x14 inputs and 240x2 logits")
    widths = recipe["widths"]
    if widths != [8, 16]:
        raise ValueError("The bounded fixture requires widths [8, 16]")
    if type(recipe["seed"]) is not int or not 0 <= recipe["seed"] < 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer with room for calibration seed")
    if type(recipe["calibration_samples"]) is not int or not 1 <= recipe["calibration_samples"] <= 256:
        raise ValueError("calibration_samples must be in [1, 256]")
    params = TcnParams.model_validate(recipe["tcn"])
    if set(recipe["tcn"]) - set(TcnParams.model_fields):
        raise ValueError("Unknown TCN controls")
    if params.block_type != "sm" or len(params.blocks) != 4:
        raise ValueError("Expected four small TCN blocks")
    if any(set(block) - set(TcnBlockParams.model_fields) for block in recipe["tcn"]["blocks"]):
        raise ValueError("Unknown TCN block controls")
    for block in params.blocks:
        if block.se_ratio != 4 or block.depth != 1 or block.branch != 1 or block.norm != "batch":
            raise ValueError("Expected SE ratio 4, depth/branch 1 and batch normalization")
        if block.activation != "relu6" or block.ex_ratio != 1 or block.dropout is not None:
            raise ValueError("Expected ReLU6, expansion 1 and no dropout")
    if not params.include_top or not params.use_logits or params.output_activation is not None:
        raise ValueError("Output must retain per-point logits")
    return recipe


def build_model(recipe, width):
    keras.backend.clear_session()
    keras.utils.set_random_seed(recipe["seed"])
    config = copy.deepcopy(recipe["tcn"])
    for block in config["blocks"]:
        block["filters"] = width
    params = TcnParams.model_validate(config)
    inputs = keras.Input(shape=tuple(recipe["input_shape"]), batch_size=1, name="features")
    model = TcnModel.model_from_params(inputs, params, num_classes=recipe["num_classes"])
    if model.output_shape != (1, 240, 2):
        raise ValueError(f"Unexpected output shape {model.output_shape}")
    return model, params.model_dump(mode="json")


def fixture_inputs(recipe):
    shape = tuple(recipe["input_shape"])
    calibration = np.random.default_rng(recipe["seed"] + 1).uniform(
        -1.0, 1.0, (recipe["calibration_samples"], *shape)
    ).astype(np.float32)
    t = np.arange(shape[0], dtype=np.float32)[:, None]
    c = np.arange(1, shape[1] + 1, dtype=np.float32)[None, :]
    signal = (0.75 * np.sin(t * c * np.float32(0.03125))).astype(np.float32)
    inputs = np.stack([np.zeros(shape, np.float32), -np.ones(shape, np.float32),
                       np.ones(shape, np.float32), signal])
    if set(map(array_hash, calibration)) & set(map(array_hash, inputs)):
        raise ValueError("Calibration and golden inputs overlap")
    return calibration, inputs


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
        operators.append({"name": op_names[opcode], "version": code.Version(),
                          "inputs": inputs, "outputs": outputs,
                          "input_dtypes": [tensors[x]["dtype"] for x in inputs],
                          "output_dtypes": [tensors[x]["dtype"] for x in outputs]})
    if precision == "INT8" and any(t["dtype"].startswith("FLOAT") for t in tensors):
        raise ValueError("Fully INT8 export contains floating-point tensors")
    return {"tensors": tensors, "operators": operators,
            "compute_contract": "INT8 activations/weights with integer bias/shape operands" if precision == "INT8"
            else "FP32 data operands with integer shape operands",
            "target_accumulator_and_dispatch": "not measured"}


def quantize(inputs, detail):
    if detail["dtype"] == np.float32:
        return inputs.copy()
    scale, zero = detail["quantization"]
    if detail["dtype"] != np.int8 or scale <= 0:
        raise ValueError("Expected per-tensor INT8 input quantization")
    return np.clip(np.rint(inputs / scale + zero), -128, 127).astype(np.int8)


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
    files = [Path(__file__).resolve(), Path(__file__).with_name("recipe.json")]
    files += sorted((root / "helia_edge").rglob("*.py"))
    files += [root / "pyproject.toml", root / "uv.lock"]
    dependencies = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()))
    return {"git_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "files": {str(p.relative_to(root)): sha256(p.read_bytes()) for p in files},
            "python": platform.python_version(), "dependencies": dependencies,
            "dependencies_sha256": sha256(json.dumps(dependencies, sort_keys=True).encode())}


def generate(recipe_path, output):
    recipe = read_recipe(recipe_path)
    output.mkdir(parents=True, exist_ok=False)
    calibration, inputs = fixture_inputs(recipe)
    write_json(output / "recipe.json", recipe)
    np.save(output / "calibration.npy", calibration, allow_pickle=False)
    np.save(output / "held_out.npy", inputs, allow_pickle=False)
    manifest = {"schema_version": 1, "kind": "synthetic performance fixture; no trained task-quality claim",
                "state": "stateless same-padding whole windows; no streaming claim",
                "source": provenance(), "recipe_sha256": sha256((output / "recipe.json").read_bytes()),
                "calibration_sample_hashes": list(map(array_hash, calibration)),
                "held_out_sample_hashes": list(map(array_hash, inputs)),
                "golden_cases": ["zero", "negative_unit_limit", "positive_unit_limit", "deterministic_signal"],
                "oracle": {"runtime": "ai-edge-litert", "version": importlib.metadata.version("ai-edge-litert"),
                           "resolver": "BUILTIN_REF", "threads": 1}, "exports": []}
    for width in recipe["widths"]:
        model, config = build_model(recipe, width)
        weights = model.get_weights()
        weight_hashes = list(map(array_hash, weights))
        np.savez(output / f"w{width}-weights.npz", **{f"weight_{i}": w for i, w in enumerate(weights)})
        write_json(output / f"w{width}-config.json", config)
        keras_reference = np.concatenate([model(x[None], training=False).numpy() for x in inputs])
        np.save(output / f"w{width}-keras.npy", keras_reference, allow_pickle=False)
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
            if inp["shape"].tolist() != [1, 240, 14] or out["shape"].tolist() != [1, 240, 2]:
                raise ValueError("Export shape mismatch")
            encoded = quantize(inputs, inp)
            golden = infer(interpreter, encoded)
            if precision == "FP32":
                np.testing.assert_allclose(golden, keras_reference, rtol=1e-5, atol=1e-5)
            if weight_hashes != list(map(array_hash, model.get_weights())):
                raise ValueError("Conversion changed the source weights")
            np.savez(output / f"{stem}-goldens.npz", inputs=encoded, outputs=golden)
            manifest["exports"].append({"width": width, "precision": precision,
                "model": f"{stem}.tflite", "model_sha256": sha256(content), "model_bytes": len(content),
                "goldens": f"{stem}-goldens.npz", "graph": f"{stem}-graph.json",
                "config": f"w{width}-config.json", "weights": f"w{width}-weights.npz",
                "weight_array_hashes": weight_hashes, "parameter_count": model.count_params(),
                "input": tensor_info(inp), "output": tensor_info(out),
                "output_bytes": int(np.prod(out["shape"])) * np.dtype(out["dtype"]).itemsize,
                "fp32_keras_max_abs_error": float(np.max(np.abs(golden - keras_reference)))
                if precision == "FP32" else None,
                "target_admission": "not run; full-output transport required"})
    manifest["files"] = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    verify(output)
    return manifest


def verify(output):
    """Check retained bytes and replay every full-output golden with the recorded oracle."""
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["oracle"]["version"] != importlib.metadata.version("ai-edge-litert"):
        raise ValueError("Oracle version mismatch")
    for name, expected in manifest["files"].items():
        if sha256((output / name).read_bytes()) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    if len(manifest["exports"]) != 4 or {(e["width"], e["precision"]) for e in manifest["exports"]} != {
        (8, "FP32"), (8, "INT8"), (16, "FP32"), (16, "INT8")
    }:
        raise ValueError("Expected four distinct width/precision exports")
    if sha256((output / "recipe.json").read_bytes()) != manifest["recipe_sha256"]:
        raise ValueError("Recipe hash mismatch")
    calibration = np.load(output / "calibration.npy", allow_pickle=False)
    held_out = np.load(output / "held_out.npy", allow_pickle=False)
    cal_hashes, held_hashes = list(map(array_hash, calibration)), list(map(array_hash, held_out))
    if cal_hashes != manifest["calibration_sample_hashes"] or held_hashes != manifest["held_out_sample_hashes"]:
        raise ValueError("Sample hash mismatch")
    if set(cal_hashes) & set(held_hashes):
        raise ValueError("Calibration and golden inputs overlap")
    for export in manifest["exports"]:
        content = (output / export["model"]).read_bytes()
        if sha256(content) != export["model_sha256"]:
            raise ValueError("Model hash mismatch")
        graph = graph_info(content, export["precision"])
        if graph != json.loads((output / export["graph"]).read_text()):
            raise ValueError("Graph metadata mismatch")
        interpreter = runtime(content)
        inp, = interpreter.get_input_details()
        out, = interpreter.get_output_details()
        if tensor_info(inp) != export["input"] or tensor_info(out) != export["output"]:
            raise ValueError("Interface metadata mismatch")
        with np.load(output / export["weights"], allow_pickle=False) as weights:
            hashes = [array_hash(weights[f"weight_{i}"]) for i in range(len(weights.files))]
        if hashes != export["weight_array_hashes"]:
            raise ValueError("Weight hash mismatch")
        with np.load(output / export["goldens"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["inputs"], quantize(held_out, inp))
            actual = infer(interpreter, data["inputs"])
            np.testing.assert_array_equal(actual, data["outputs"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("recipe.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Verify and replay an existing retained fixture")
    args = parser.parse_args()
    manifest = verify(args.output) if args.verify else generate(args.recipe, args.output)
    print(json.dumps({"exports": len(manifest["exports"]), "manifest": str(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
