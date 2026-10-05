"""Retain one seeded FP32 architecture fixture with its export record, not official trained weights."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import keras
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from helia_edge.export import ExportOptions, export
from helia_edge.export.result import environment_record, unidentified_install
from helia_edge.models import MlperfTinyParams, ModelSpec, build

SPECS = {a: ModelSpec(params=MlperfTinyParams(architecture=a)) for a in ("kws", "vww", "resnet", "ad")}


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
    """Export the seeded model to ``output`` (``model.tflite``, ``model.weights.h5``, ``record.json``) with
    ``goldens.npz`` (the diagnostic cases run with LiteRT's reference kernels, their amplitude and the seed),
    ``references.json`` (the pinned upstream sources) and the MLPerf Tiny licenses. ``record.json`` is
    written last.

    Returns:
        Path: The ``record.json``.
    """
    if keras.backend.backend() != "tensorflow":
        raise ValueError("FP32 export requires the TensorFlow Keras backend")
    if name not in SPECS:
        raise ValueError(f"Unknown model {name}")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    output.mkdir(parents=True, exist_ok=False)
    install = environment_record()
    if not install.identified:  # the record then cannot name the code that exported
        print(f"warning: {unidentified_install(install)}", file=sys.stderr)
    keras.utils.set_random_seed(seed)
    model = build(SPECS[name])
    inputs, expected, amplitude = diagnostic_cases(model)
    result = export(
        model, precision="fp32", io_dtype="float32", spec=SPECS[name], options=ExportOptions(mode="concrete")
    )
    interpreter = Interpreter(
        model_content=result.content, num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    (inp,) = interpreter.get_input_details()
    (out,) = interpreter.get_output_details()
    assert inp["dtype"] == out["dtype"] == np.float32
    actual = []
    for x in inputs:
        interpreter.set_tensor(inp["index"], x[None])
        interpreter.invoke()
        actual.append(interpreter.get_tensor(out["index"])[0])
    actual = np.stack(actual)
    check_outputs(actual, expected)
    np.savez(
        output / "goldens.npz", inputs=inputs, outputs=actual, keras_outputs=expected, amplitude=amplitude, seed=seed
    )
    source = Path(__file__).resolve().parents[2] / "helia_edge/models/mlperf_tiny.py"
    shutil.copyfile(Path(__file__).with_name("references.json"), output / "references.json")
    for license_file in (source.parent / "licenses").glob("mlperf-tiny-*.txt"):
        shutil.copyfile(license_file, output / license_file.name)
    return result.write(output)  # record.json last: it marks a complete export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SPECS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    print(json.dumps({"model": args.model, "record": str(generate(args.model, args.output, args.seed))}))
