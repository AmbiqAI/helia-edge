"""Train with the Torch backend, export with the TensorFlow backend, and compare.

train  (KERAS_BACKEND=torch):      fit seeded models briefly; save weights, params and Torch outputs.
export (KERAS_BACKEND=tensorflow): rebuild from params, load weights, compare Keras and LiteRT outputs;
                                   --report FILE also writes the differences and bounds as JSON.
run: both halves in separate processes, given one Python for each backend.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

SEED = 20261001
SAMPLES = 16

# Largest |output difference| accepted against the Torch-trained model's own outputs. Observed with
# TensorFlow 2.21, Torch 2.14 and ai-edge-litert 2.2: Keras <= 1.1e-7, LiteRT fp32 <= 9e-8 and a8w8
# (calibrated on the compared inputs) <= 9e-3. The float bounds keep about a 90x margin; the a8w8 bound
# keeps 2x and rejects untrained weights for the TCN and MiniResNet models (errors >= 0.4), but not for
# KWS, whose untrained a8w8 error is about 5e-3.
TOLERANCE = {"keras": 1e-5, "fp32": 1e-5, "a8w8": 2e-2}
# Fresh (untrained) weights must differ from the trained outputs by more than this (10x the Keras
# bound; observed >= 4e-3), so the data and training are strong enough to expose a skipped load.
SENSITIVITY = 1e-4


def models():
    """Name -> (builder from params JSON, params JSON, input shape without batch)."""
    from helia_edge.models import MiniResNetV1Params, MlperfTinyParams, TcnParams, compact_tcn_params
    from helia_edge.models.miniresnet import build as miniresnet_build
    from helia_edge.models.mlperf_tiny import build as mlperf_build
    from helia_edge.models.tcn import build as tcn_build

    tcn = compact_tcn_params(filters=8)
    miniresnet = MiniResNetV1Params(base_filters=8, stacks=1, pooling="avg")
    kws = MlperfTinyParams(architecture="kws")
    return {
        "compact-tcn": (
            lambda p: tcn_build(TcnParams.model_validate({**p, "num_classes": 3}), (64, 14)),
            tcn.model_dump(mode="json"),
            (64, 14),
        ),
        "miniresnet-v1": (
            lambda p: miniresnet_build(MiniResNetV1Params.model_validate({**p, "num_classes": 4}), (32, 20, 1)),
            miniresnet.model_dump(mode="json"),
            (32, 20, 1),
        ),
        "mlperf-kws": (
            lambda p: mlperf_build(MlperfTinyParams.model_validate(p)),
            kws.model_dump(mode="json"),
            (49, 10, 1),
        ),
    }


def data(name, shape):
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode())])
    return rng.standard_normal((SAMPLES, *shape)).astype(np.float32)


def train(output: Path):
    import keras

    if keras.backend.backend() != "torch":
        raise SystemExit(f"train needs KERAS_BACKEND=torch, not {keras.backend.backend()!r}")
    output.mkdir(parents=True, exist_ok=True)
    reference = {}
    for name, (build, params, shape) in models().items():
        keras.utils.set_random_seed(SEED)
        model = build(params)
        x = data(name, shape)
        y = np.asarray(keras.ops.convert_to_numpy(model(x[:1], training=False)))
        targets = np.random.default_rng(SEED).standard_normal((SAMPLES, *y.shape[1:])).astype(np.float32)
        model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
        model.fit(x, targets, epochs=3, batch_size=4, shuffle=False, verbose=0)
        model.save_weights(output / f"{name}.weights.h5")
        (output / f"{name}.params.json").write_text(json.dumps(params, sort_keys=True))
        reference[name] = keras.ops.convert_to_numpy(model(x, training=False))
    np.savez(output / "torch-outputs.npz", **reference)
    print(f"trained {len(reference)} models with {keras.backend.backend()} into {output}")


def export(source: Path, report: Path | None):
    import keras

    from helia_edge.export import ExportSpec, LiteRTRunner, export_model

    if keras.backend.backend() != "tensorflow":
        raise SystemExit(f"export needs KERAS_BACKEND=tensorflow, not {keras.backend.backend()!r}")
    torch_outputs = np.load(source / "torch-outputs.npz")
    results, failures = {}, []
    for name, (build, _, shape) in models().items():
        params = json.loads((source / f"{name}.params.json").read_text())
        x, expected = data(name, shape), torch_outputs[name]
        keras.utils.set_random_seed(SEED + 1)
        model = build(params)
        fresh = float(np.abs(keras.ops.convert_to_numpy(model(x, training=False)) - expected).max())
        model.load_weights(source / f"{name}.weights.h5")
        row = {
            "fresh_weights": fresh,
            "keras": float(np.abs(keras.ops.convert_to_numpy(model(x, training=False)) - expected).max()),
        }
        # LiteRT runs on its reference kernels, the convention for benchmark goldens. Its XNNPACK delegate
        # (ai-edge-litert 2.2) cannot prepare the compact TCN at a8w8, with integer or float I/O.
        for precision, io_dtype in (("fp32", "float32"), ("a8w8", "int8")):
            spec = ExportSpec(precision=precision, io_dtype=io_dtype, mode="concrete")
            result = export_model(model, spec, x if precision == "a8w8" else None)
            y = LiteRTRunner(result.content, reference_kernels=True).predict(x)
            if y.shape != expected.shape:
                failures.append(f"{name}: {precision} output shape {y.shape} differs from {expected.shape}")
                y = np.full(expected.shape, np.inf, np.float32)
            row[precision] = float(np.abs(y - expected).max())
        results[name] = row
        if fresh <= SENSITIVITY:
            failures.append(f"{name}: fresh weights already match within {fresh:.2e}; the check is not sensitive")
        failures += [f"{name}: {k} differs by {row[k]:.2e} > {t:.0e}" for k, t in TOLERANCE.items() if row[k] > t]
    print(json.dumps(results, indent=1))
    if report:
        report.write_text(
            json.dumps({"tolerance": TOLERANCE, "sensitivity": SENSITIVITY, "results": results}, indent=1)
        )
    if failures:
        raise SystemExit("cross-backend export failed:\n" + "\n".join(failures))


def run(torch_python: str, tensorflow_python: str):
    with tempfile.TemporaryDirectory() as directory:
        for backend, python, args in (
            ("torch", torch_python, ["train", "--output", directory]),
            ("tensorflow", tensorflow_python, ["export", "--input", directory]),
        ):
            env = dict(os.environ, KERAS_BACKEND=backend)
            subprocess.run([python, __file__, *args], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train").add_argument("--output", type=Path, required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--input", type=Path, required=True)
    exporter.add_argument("--report", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("--torch-python", default=sys.executable)
    runner.add_argument("--tensorflow-python", default=sys.executable)
    args = parser.parse_args()
    if args.command == "train":
        train(args.output)
    elif args.command == "export":
        export(args.input, args.report)
    else:
        run(args.torch_python, args.tensorflow_python)


if __name__ == "__main__":
    main()
