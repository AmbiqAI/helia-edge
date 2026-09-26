"""Retain one seeded FP32 architecture fixture, not official trained weights."""

import argparse
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

from helia_edge.converters.litert import ConversionType, LiteRTKerasConverter, QuantizationType
from helia_edge.models.mlperf_tiny import mlperf_tiny_ad, mlperf_tiny_kws, mlperf_tiny_resnet, mlperf_tiny_vww

BUILDERS = {"kws": mlperf_tiny_kws, "vww": mlperf_tiny_vww, "resnet": mlperf_tiny_resnet, "ad": mlperf_tiny_ad}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic_cases(model):
    """Amplify synthetic inputs until initialized outputs expose signal propagation."""
    shape = tuple(model.input_shape[1:])
    signal = np.sin(np.arange(np.prod(shape), dtype=np.float32) * np.float32(0.03125)).reshape(shape)
    zero = np.zeros(shape, np.float32)
    baseline = model(zero[None], training=False).numpy()[0]
    for exponent in range(25):
        amplitude = float(10**exponent)
        inputs = np.stack([zero, signal * np.float32(amplitude)])
        expected = np.stack([baseline, model(inputs[1:2], training=False).numpy()[0]])
        if np.isfinite(expected).all() and np.max(np.abs(expected[1] - baseline)) >= 0.01:
            return inputs, expected, amplitude
    raise ValueError("Initialized model has no numerically discriminating diagnostic signal")


def check_outputs(actual, expected):
    if not np.isfinite(expected).all() or np.max(np.abs(expected[1] - expected[0])) < 0.01:
        raise ValueError("Reference cases cannot reject an input-independent output")
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def generate(name, output, seed=20260926):
    if keras.backend.backend() != "tensorflow":
        raise ValueError("FP32 export requires the TensorFlow Keras backend")
    if name not in BUILDERS:
        raise ValueError(f"Unknown model {name}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    output.mkdir(parents=True, exist_ok=False)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = BUILDERS[name]()
    inputs, expected, amplitude = diagnostic_cases(model)
    model.save(output / "model.keras")
    (output / "config.json").write_text(model.to_json(indent=2) + "\n")
    converter = LiteRTKerasConverter(model)
    try:
        content = converter.convert(inputs, quantization=QuantizationType.FP32, mode=ConversionType.CONCRETE)
    finally:
        converter.cleanup()
    (output / "model.tflite").write_bytes(content)
    interpreter = Interpreter(model_content=content, num_threads=1,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    assert inp["dtype"] == out["dtype"] == np.float32
    actual = []
    for x in inputs:
        interpreter.set_tensor(inp["index"], x[None])
        interpreter.invoke()
        actual.append(interpreter.get_tensor(out["index"])[0])
    actual = np.stack(actual)
    check_outputs(actual, expected)
    np.savez(output / "goldens.npz", inputs=inputs, outputs=actual, keras_outputs=expected)
    root = Path(__file__).resolve().parents[2]
    source = root / "helia_edge/models/mlperf_tiny.py"
    shutil.copyfile(Path(__file__).with_name("references.json"), output / "references.json")
    for path in (source.parent / "licenses").glob("mlperf-tiny-*.txt"):
        shutil.copyfile(path, output / path.name)
    dependencies = {key: importlib.metadata.version(key) for key in
                    ("keras", "tensorflow", "ai-edge-litert", "numpy", "helia-edge")}
    manifest = {"model": name, "scale": 1.0, "seed": seed,
                "kind": "synthetic architecture fixture; no official trained weights/accuracy/compliance claim",
                "source_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": digest(source), "generator_sha256": digest(Path(__file__)),
                "reference": "references.json", "lock_sha256": digest(root / "uv.lock"), "precision": "FP32", "dependencies": dependencies,
                "python": platform.python_version(), "oracle": "ai-edge-litert BUILTIN_REF; one thread; no delegates",
                "cases": ["zero", "amplified_deterministic_signal"], "signal_amplitude": amplitude,
                "diagnostic_only": "amplitude selected for initialized-output discrimination; not real data", "preprocessing": "none; synthetic feature-domain inputs",
                "max_abs_keras_error": float(np.max(np.abs(actual - expected))),
                "input": {"name": inp["name"], "index": int(inp["index"]), "shape": inp["shape"].tolist(),
                          "dtype": "float32", "bytes": int(np.prod(inp["shape"])) * 4},
                "output": {"name": out["name"], "index": int(out["index"]), "shape": out["shape"].tolist(),
                           "dtype": "float32", "bytes": int(np.prod(out["shape"])) * 4},
                "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=BUILDERS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    result = generate(args.model, args.output, args.seed)
    print(json.dumps({"model": args.model, "manifest": str(args.output / "manifest.json"),
                      "max_abs_keras_error": result["max_abs_keras_error"]}))
