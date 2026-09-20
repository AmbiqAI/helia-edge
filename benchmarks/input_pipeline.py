"""Synthetic stage timing; run from the repo with the TensorFlow extra installed.

CUDA_VISIBLE_DEVICES=-1 KERAS_BACKEND=tensorflow python benchmarks/input_pipeline.py
This is a correctness/timing baseline, not evidence of a real-data speedup.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import keras
import numpy as np
import tensorflow as tf

from helia_edge.utils import create_interleaved_dataset_from_generator


def timed(fn, repetitions):
    measurements = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        measurements.append(time.perf_counter() - start)
    return {"seconds": measurements, "median_seconds": float(np.median(measurements))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=1025)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.examples, args.batch_size, args.workers, args.repetitions) < 1:
        parser.error("all counts must be positive")
    keras.utils.set_random_seed(0)
    schedule = list(range(args.examples))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "features.npy"

        def prepare():
            # Synthetic deterministic preparation, not source decoding.
            data = np.arange(args.examples * 20, dtype=np.float32).reshape(args.examples, 4, 5) / 1000
            np.save(path, data)

        start = time.perf_counter()
        prepare()
        cold = time.perf_counter() - start
        source_bytes = path.stat().st_size

        def samples(ids):
            # Each iterator opens its own read-only mapping.
            data = np.load(path, mmap_mode="r")
            for index in ids:
                yield data[index], np.float32(index % 2)

        spec = (tf.TensorSpec((4, 5), tf.float32), tf.TensorSpec((), tf.float32))

        def dataset(mode):
            ds = (
                create_interleaved_dataset_from_generator(
                    samples,
                    iter,
                    schedule,
                    spec,
                    num_workers=args.workers,
                    stream_mode=mode,
                )
                .batch(args.batch_size, drop_remainder=False)
                .prefetch(1)
            )
            options = tf.data.Options()
            options.threading.private_threadpool_size = 2
            return ds.with_options(options)

        reference = list(dataset("global").as_numpy_iterator())
        for actual, expected in zip(dataset("finite").as_numpy_iterator(), reference, strict=True):
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)

        def model():
            m = keras.Sequential([keras.Input((4, 5)), keras.layers.Flatten(), keras.layers.Dense(1)])
            m.compile(optimizer=keras.optimizers.SGD(0.001), loss="mse", jit_compile=False)
            return m

        resident_model = model()
        initial_weights = resident_model.get_weights()
        resident = [(tf.convert_to_tensor(x), tf.convert_to_tensor(y)) for x, y in reference]
        # First step includes compile/optimizer setup and is reported separately.
        start = time.perf_counter()
        resident_model.train_on_batch(*resident[0])
        startup = time.perf_counter() - start

        def train_batches(m, batches):
            for x, y in batches:
                # train_on_batch returns host loss values, synchronizing each CPU step.
                m.train_on_batch(x, y)

        train_batches(resident_model, resident)  # Also warm the final partial-batch shape.
        report = {
            "kind": "synthetic CPU baseline; no real-data throughput claim",
            "environment": {name: importlib.metadata.version(name) for name in ("keras", "tensorflow", "numpy")},
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "configuration": {k: v for k, v in vars(args).items() if k != "output"},
            "sampling": "finite, ordered, no replacement, no balancing, retain partial batch",
            "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
            "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prepared_bytes": source_bytes,
            "cold_synthetic_preparation_seconds": cold,
            "first_model_step_seconds": startup,
            "model_warmup": "one first step plus one full epoch (including partial batch)",
            "resident_batches": timed(lambda: train_batches(resident_model, resident), args.repetitions),
            "paths": {},
        }
        for mode in ("global", "finite"):
            ds = dataset(mode)

            def load():
                count = sum(len(x) for x, _ in ds.as_numpy_iterator())
                assert count == args.examples

            load()  # Warm filesystem/iterator caches before timing.
            warm = timed(load, args.repetitions)
            warm["examples_per_second"] = args.examples / warm["median_seconds"]
            m = model()
            m.set_weights(initial_weights)
            m.train_on_batch(*resident[0])
            train_batches(m, resident)
            report["paths"][mode] = {
                "warm_loader": warm,
                "end_to_end": timed(lambda: train_batches(m, ds), args.repetitions),
            }
        # Linux ru_maxrss is KiB; this high-water mark covers all stages together.
        report["process_peak_rss_platform_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["limitations"] = (
            "No real decoding, accelerator utilization/transfer timing or per-stage memory measurement."
        )
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            print(output)


if __name__ == "__main__":
    main()
