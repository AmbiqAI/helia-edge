"""Synthetic CPU input timing; see docs/input-pipeline.md for the protocol."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
import platform as platform_info
import resource
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import keras
import numpy as np
import tensorflow as tf

from helia_edge.utils import StreamMode, create_interleaved_dataset_from_generator

if TYPE_CHECKING:
    from helia_edge._typing import Array


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    examples: int = 1025
    batch_size: int = 32
    workers: int = 4
    repetitions: int = 3

    def __post_init__(self) -> None:
        if min(self.examples, self.batch_size, self.workers, self.repetitions) < 1:
            raise ValueError("all counts must be positive")


@dataclass(frozen=True, slots=True)
class Timing:
    seconds: list[float]
    median_seconds: float
    examples_per_second: float | None = None


@dataclass(frozen=True, slots=True)
class PipelineTiming:
    warm_loader: Timing
    end_to_end: Timing


@dataclass(slots=True)
class BenchmarkReport:
    configuration: BenchmarkConfig
    schedule_sha256: str
    prepared_sha256: str
    prepared_bytes: int
    cold_synthetic_preparation_seconds: float
    first_model_step_seconds: float
    resident_batches: Timing
    paths: dict[StreamMode, PipelineTiming] = field(default_factory=dict)
    process_peak_rss_platform_units: int = 0
    environment: dict[str, str] = field(
        default_factory=lambda: {name: importlib.metadata.version(name) for name in ("keras", "tensorflow", "numpy")}
    )
    python: str = field(default_factory=platform_info.python_version)
    platform: str = field(default_factory=platform_info.platform)
    processor: str = field(default_factory=platform_info.processor)
    logical_cpus: int | None = field(default_factory=os.cpu_count)
    kind: str = "synthetic CPU baseline; no real-data throughput claim"
    sampling: str = "finite, ordered, no replacement, no balancing, retain partial batch"
    model_warmup: str = "one first step plus one full epoch (including partial batch)"
    limitations: str = "No real decoding, accelerator utilization/transfer timing or per-stage memory measurement."


def timed(fn: Callable[[], None], repetitions: int, examples: int | None = None) -> Timing:
    measurements = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        measurements.append(time.perf_counter() - start)
    median = float(np.median(measurements))
    return Timing(measurements, median, examples / median if examples is not None else None)


def make_model() -> keras.Model:
    model = keras.Sequential([keras.Input((4, 5)), keras.layers.Flatten(), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.SGD(0.001), loss="mse", jit_compile=False)
    return model


def train_batches(model: keras.Model, batches: Iterable[tuple[Array, Array]]) -> None:
    for x, y in batches:
        # Host loss values synchronize each CPU step.
        model.train_on_batch(x, y)


def benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    keras.utils.set_random_seed(0)
    schedule = list(range(config.examples))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "features.npy"
        start = time.perf_counter()
        data = np.arange(config.examples * 20, dtype=np.float32).reshape(config.examples, 4, 5) / 1000
        np.save(path, data)
        cold = time.perf_counter() - start

        def samples(ids: Iterator[int]) -> Iterator[tuple[np.ndarray, np.float32]]:
            # Each iterator owns its read-only mapping.
            prepared = np.load(path, mmap_mode="r")
            for index in ids:
                yield prepared[index], np.float32(index % 2)

        spec = (tf.TensorSpec((4, 5), tf.float32), tf.TensorSpec((), tf.float32))

        def dataset(mode: StreamMode) -> tf.data.Dataset:
            ds = (
                create_interleaved_dataset_from_generator(
                    samples,
                    iter,
                    schedule,
                    spec,
                    num_workers=config.workers,
                    stream_mode=mode,
                )
                .batch(config.batch_size, drop_remainder=False)
                .prefetch(1)
            )
            options = tf.data.Options()
            options.threading.private_threadpool_size = 2
            return ds.with_options(options)

        reference = list(dataset(StreamMode.GLOBAL).as_numpy_iterator())
        for actual, expected in zip(dataset(StreamMode.FINITE).as_numpy_iterator(), reference, strict=True):
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left, right)

        model = make_model()
        initial_weights = model.get_weights()
        resident = [(tf.convert_to_tensor(x), tf.convert_to_tensor(y)) for x, y in reference]
        start = time.perf_counter()
        model.train_on_batch(*resident[0])
        startup = time.perf_counter() - start
        train_batches(model, resident)  # Warm the partial-batch shape too.
        report = BenchmarkReport(
            configuration=config,
            schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
            prepared_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            prepared_bytes=path.stat().st_size,
            cold_synthetic_preparation_seconds=cold,
            first_model_step_seconds=startup,
            resident_batches=timed(lambda: train_batches(model, resident), config.repetitions),
        )
        for mode in StreamMode:
            ds = dataset(mode)

            def load() -> None:
                count = sum(len(x) for x, _ in ds.as_numpy_iterator())
                assert count == config.examples

            load()
            warm = timed(load, config.repetitions, examples=config.examples)
            model = make_model()
            model.set_weights(initial_weights)
            model.train_on_batch(*resident[0])
            train_batches(model, resident)
            report.paths[mode] = PipelineTiming(
                warm_loader=warm,
                end_to_end=timed(lambda: train_batches(model, ds), config.repetitions),
            )
        # Linux reports KiB; this high-water mark covers all stages.
        report.process_peak_rss_platform_units = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=1025)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = BenchmarkConfig(args.examples, args.batch_size, args.workers, args.repetitions)
    except ValueError as exc:
        parser.error(str(exc))
    output = json.dumps(asdict(benchmark(config)), indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
