"""TensorFlow dataset adapters and preprocessing bounds."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .sampling import StreamMode
from numbers import Integral

import keras
import tensorflow as tf
import numpy.typing as npt


def parse_factor(
    param: float | tuple[float | None, float] | list[float | None],
    min_value: float | None = 0.0,
    max_value: float | None = 1.0,
    param_name: str = "factor",
) -> tuple[float, float]:
    """Normalize scalar or paired bounds and validate their range."""
    low, high = (min_value, param) if isinstance(param, (float, int)) else param
    if high is None:
        raise ValueError(f"{param_name} requires an upper bound")
    low = high if low is None else low
    if low > high:
        raise ValueError(f"{param_name}[0] must be <= {param_name}[1]; got {param}")
    if (min_value is not None and low < min_value) or (max_value is not None and high > max_value):
        raise ValueError(f"{param_name} must be inside [{min_value}, {max_value}]; got {param}")
    return low, high


def convert_inputs_to_tf_dataset(x=None, y=None, sample_weight=None, batch_size=None):
    """Convert inputs to tf.data.Dataset."""

    # Unpack if passed as tuple
    if isinstance(x, tuple):
        tupled = x
        x = tupled[0]
        y = tupled[1] if len(tupled) > 1 else None
        sample_weight = tupled[2] if len(tupled) > 2 else None

    if sample_weight is not None:
        raise ValueError("Contrastive trainers do not yet support `sample_weight`.")

    if isinstance(x, tf.data.Dataset):
        if y is not None or batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not "
                "provide a value for `y` or `batch_size`. "
                "Got `y={y}`, `batch_size={batch_size}`."
            )
        return x

    # batch_size defaults to 32, as it does in fit().
    batch_size = batch_size or 32
    # Parse inputs
    inputs = x
    if y is not None:
        inputs = (x, y)

    # Construct tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset


def create_interleaved_dataset_from_generator[T, K](
    data_generator: Callable[[Iterator[T]], Iterable[K]],
    id_generator: Callable[[list[T]], Iterator[T]],
    ids: list[T],
    spec: tf.TensorSpec | tuple[tf.TensorSpec, ...] | dict[str, tf.TensorSpec],
    preprocess: Callable[[K], K] | None = None,
    num_workers: int = 4,
    *,
    stream_mode: StreamMode | str = StreamMode.GLOBAL,
    deterministic: bool = True,
) -> tf.data.Dataset:
    """Adapt caller-owned schedules to tf.data without changing sample weights.

    GLOBAL preserves one finite/repeated stream. FINITE partitions terminating,
    partition-independent generators; deterministic mode preserves partition order.
    num_workers counts generators, not processes. See docs/input-pipeline.md.
    """

    if isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    try:
        stream_mode = StreamMode(stream_mode)
    except ValueError as exc:
        raise ValueError(f"stream_mode must be one of {list(StreamMode)}") from exc
    # Snapshot IDs so caller mutation cannot change later epoch enumeration.
    ids = list(ids)

    def split_generator(split_ids: list[T]) -> tf.data.Dataset:
        def ds_gen() -> Iterator[K]:
            split_id_generator = id_generator(list(split_ids))
            samples = iter(data_generator(split_id_generator))
            return map(preprocess, samples) if preprocess is not None else samples

        return tf.data.Dataset.from_generator(
            ds_gen,
            output_signature=spec,
        )

    if not ids:
        return tf.data.Dataset.from_generator(lambda: iter(()), output_signature=spec)
    if stream_mode == StreamMode.GLOBAL:
        return split_generator(ids)

    num_workers = min(num_workers, len(ids))
    size, remainder = divmod(len(ids), num_workers)
    ds_splits = []
    start = 0
    for i in range(num_workers):
        end = start + size + (i < remainder)
        ds_splits.append(split_generator(ids[start:end]))
        start = end

    if deterministic:
        ds = ds_splits[0]
        for split in ds_splits[1:]:
            ds = ds.concatenate(split)
        return ds

    return tf.data.Dataset.from_tensor_slices(ds_splits).interleave(
        lambda x: x,
        cycle_length=num_workers,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def create_dataset_from_data(x: npt.NDArray, y: npt.NDArray, spec: tuple[tf.TensorSpec, ...]) -> tf.data.Dataset:
    """Helper function to create dataset from static data

    Args:
        x (npt.NDArray): Numpy data
        y (npt.NDArray): Numpy labels

    Returns:
        tf.data.Dataset: Dataset
    """
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)))


def get_output_signature(
    outputs: keras.KerasTensor | npt.NDArray | tuple[keras.KerasTensor | npt.NDArray],
) -> tf.TensorSpec | tuple[tf.TensorSpec, ...]:
    """Get output signature from sample outputs

    Args:
        outputs: Outputs. A tensor or tuple of tensors. Either KerasTensor, tf.Tensor, or numpy array.

    Returns:
        tf.TensorSpec: Tensor spec
    """
    if isinstance(outputs, tuple):
        sig = []
        for output in outputs:
            output = keras.ops.convert_to_tensor(output)
            sig.append(tf.TensorSpec(shape=output.shape, dtype=output.dtype))
        sig = tuple(sig)
    else:
        output = keras.ops.convert_to_tensor(outputs)
        sig = tf.TensorSpec(shape=output.shape, dtype=output.dtype)
    return sig


def get_output_signature_from_fn(
    fn: Callable[..., keras.KerasTensor], *args
) -> tf.TensorSpec | tuple[tf.TensorSpec, ...]:
    """Get output signature from a function

    Args:
        fn (Callable[..., tf.Tensor]): Function

    Returns:
        tf.TensorSpec: Tensor spec
    """
    return get_output_signature(outputs=fn(*args))


def get_output_signature_from_gen(
    gen: Callable[..., Iterator[Any]], *args: Any
) -> tf.TensorSpec | tuple[tf.TensorSpec, ...]:
    """Get output signature from a generator

    Args:
        gen: Generator factory

    Returns:
        tf.TensorSpec: Tensor spec
    """
    return get_output_signature(outputs=next(gen(*args)))
