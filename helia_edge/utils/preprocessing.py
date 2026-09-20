"""
# Preprocessing Utility API

This module provides utility functions for preprocessing data.

Functions:
    parse_factor: Parse factor
    convert_inputs_to_tf_dataset: Convert inputs to tf.data.Dataset
    create_interleaved_dataset_from_generator: Create interleaved dataset from generator
    create_dataset_from_data: Create dataset from data
    get_output_signature: Get output signature
    get_output_signature_from_fn: Get output signature from function
    get_output_signature_from_gen: Get output signature from generator

"""

from typing import TypeVar, Callable, Generator, Literal
from numbers import Integral

import keras
import tensorflow as tf
import numpy.typing as npt

T, K = TypeVar("T"), TypeVar("K")


def parse_factor(
    param: T | tuple[T, T], min_value: float = 0.0, max_value: float = 1.0, param_name: str = "factor"
) -> tuple[T, T]:
    if isinstance(param, (float, int)):
        param = (min_value, param)
    # END IF

    if param[0] is None:
        param[0] = param[1]
    # END IF

    if param[0] > param[1]:
        raise ValueError(
            f"`{param_name}[0] > {param_name}[1]`, `{param_name}[0]` must be "
            f"<= `{param_name}[1]`. Got `{param_name}={param}`"
        )
    if (min_value is not None and param[0] < min_value) or (max_value is not None and param[1] > max_value):
        raise ValueError(
            f"`{param_name}` should be inside of range [{min_value}, {max_value}]. Got {param_name}={param}"
        )
    # END IF

    return param[0], param[1]


def convert_inputs_to_tf_dataset(x=None, y=None, sample_weight=None, batch_size=None):
    """Convert inputs to tf.data.Dataset."""

    # Unpack if passed as tuple
    if isinstance(x, tuple):
        tupled = x
        x = tupled[0]
        y = tupled[1] if len(tupled) > 1 else None
        sample_weight = tupled[2] if len(tupled) > 2 else None
    # END IF

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


def create_interleaved_dataset_from_generator(
    data_generator: Callable[[Generator[T, None, None]], Generator[K, None, None]],
    id_generator: Callable[[list[T]], Generator[T, None, None]],
    ids: list[T],
    spec: tuple[tf.TensorSpec, tf.TensorSpec],
    preprocess: Callable[[K], K] | None = None,
    num_workers: int = 4,
    *,
    stream_mode: Literal["global", "finite"] = "global",
    deterministic: bool = True,
) -> tf.data.Dataset:
    """Adapt Python generators to a TF dataset without changing sample weights.

    ``global`` (default) invokes the generators once on all IDs, preserving their
    finite or repeated sampling law and ordering. ``num_workers`` does not split
    this stream. This correctness default replaces the old implicit partitioning
    of repeated streams, which dropped remainder IDs and changed their weights.

    ``finite`` is an explicit caller promise that each partition terminates and
    its generator semantics are partition-independent. Every input ID is assigned
    exactly once. Ordered mode concatenates contiguous partitions; unordered mode
    interleaves them and preserves full-epoch coverage, not order. Stopping early
    need not preserve sampling probabilities. Neither mode creates processes.

    Args:
        data_generator (Callable[[Generator[T, None, None]], Generator[K, None, None]]): Data generator
        id_generator (Callable[[list[T]], Generator[T, None, None]]): Id generator
        ids (list[T]): List of ids
        spec (tuple[tf.TensorSpec, tf.TensorSpec]): Tensor spec
        preprocess (Callable[[K], K] | None, optional): Preprocess function. Defaults to None.
        num_workers (int, optional): Maximum finite generator partitions, positive.
        stream_mode: Global schedule or explicitly finite partitioned streams.
        deterministic: Preserve partition order in finite mode. Global mode always
            preserves the order produced by the caller's generators.

    Returns:
        tf.data.Dataset: Dataset
    """

    if isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    if stream_mode not in ("global", "finite"):
        raise ValueError("stream_mode must be 'global' or 'finite'")
    # Snapshot IDs so caller mutation cannot change later epoch enumeration.
    ids = list(ids)

    def split_generator(split_ids: list[T]) -> tf.data.Dataset:
        """Split generator per worker"""

        def ds_gen():
            """Worker generator routine"""
            split_id_generator = id_generator(list(split_ids))
            samples = data_generator(split_id_generator)
            return map(preprocess, samples) if preprocess is not None else samples

        return tf.data.Dataset.from_generator(
            ds_gen,
            output_signature=spec,
        )

    if not ids:
        return tf.data.Dataset.from_generator(lambda: iter(()), output_signature=spec)
    if stream_mode == "global":
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


def create_dataset_from_data(x: npt.NDArray, y: npt.NDArray, spec: tuple[tf.TensorSpec]) -> tf.data.Dataset:
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
) -> tf.TensorSpec | tuple[tf.TensorSpec]:
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


def get_output_signature_from_fn(fn: Callable[..., keras.KerasTensor], *args) -> tf.TensorSpec | tuple[tf.TensorSpec]:
    """Get output signature from a function

    Args:
        fn (Callable[..., tf.Tensor]): Function

    Returns:
        tf.TensorSpec: Tensor spec
    """
    return get_output_signature(outputs=fn(*args))


def get_output_signature_from_gen(gen: Generator[T, None, None], *args) -> tf.TensorSpec | tuple[tf.TensorSpec]:
    """Get output signature from a generator

    Args:
        gen (Generator[T, None, None]): Generator

    Returns:
        tf.TensorSpec: Tensor spec
    """
    return get_output_signature(outputs=next(gen(*args)))
