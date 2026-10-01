"""Grain pipelines over a ``DataSource``."""

from collections.abc import Callable
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

from .source import DataSource

if TYPE_CHECKING:
    import grain


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _check_count(name: str, value: object, minimum: int) -> None:
    if not (isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")


def to_grain(
    source: DataSource[Any],
    *,
    seed: int | None = None,
    shuffle: bool = False,
    transform: Callable[[Any, np.random.Generator], Any] | None = None,
    batch_size: int | None = None,
    drop_remainder: bool = False,
    num_epochs: int | None = 1,
    workers: int = 0,
    worker_buffer_size: int = 1,
) -> "grain.IterDataset":
    """Read ``source`` through Grain: shuffle, repeat, transform, batch and prefetch.

    The order is source, shuffle (a new permutation each epoch), repeat, ``transform``, batch, so
    batches run across epoch boundaries. Record order and transform randomness depend only on
    ``seed``, not on ``workers``. Every ``iter()`` replays the same elements, so a Keras ``fit``
    that iterates once per epoch sees the same epoch each time: for Keras training, pass
    ``num_epochs=None`` and set ``steps_per_epoch``.

    Args:
        source: Records addressed by index.
        seed: Seed for the shuffle and for the per-record generator passed to ``transform``,
            from 0 to 2**32 - 1. Required when ``shuffle`` is set or ``transform`` is given.
        shuffle: Shuffle the records of every epoch (each of the ``num_epochs`` passes).
        transform: ``transform(record, rng) -> record``, run per record in the Grain workers;
            ``rng`` is a NumPy generator derived from ``seed`` and the record's position.
        batch_size: Stack this many records into one batch of NumPy arrays; None keeps single
            records.
        drop_remainder: Drop the last batch if it is smaller than ``batch_size``.
        num_epochs: Number of passes over ``source``; None repeats indefinitely.
        workers: Grain worker processes; 0 reads in this process.
        worker_buffer_size: Elements each worker prepares ahead.

    Returns:
        grain.IterDataset: Re-iterable; each ``iter()`` starts from the first element.

    Raises:
        ValueError: If ``seed`` is missing when needed or out of range, or a count is not a
            positive integer (``workers`` may be 0).
        ImportError: If Grain is not installed (``helia-edge[grain]``).
    """
    if seed is None and (shuffle or transform is not None):
        raise ValueError("to_grain needs a seed when shuffle is set or a transform is given")
    if seed is not None and not (_is_int(seed) and 0 <= seed < 2**32):
        raise ValueError(f"seed must be an integer from 0 to 2**32 - 1; got {seed!r}")
    if batch_size is not None:
        _check_count("batch_size", batch_size, 1)
    if num_epochs is not None:
        _check_count("num_epochs", num_epochs, 1)
    _check_count("workers", workers, 0)
    _check_count("worker_buffer_size", worker_buffer_size, 1)
    try:
        import grain
    except ModuleNotFoundError as exc:
        if exc.name != "grain":
            raise
        raise ImportError("to_grain requires Grain. Install helia-edge[grain].") from exc

    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)
    if transform is not None:
        dataset = dataset.random_map(transform, seed=seed)
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    result = dataset.to_iter_dataset()
    if workers:
        options = grain.MultiprocessingOptions(num_workers=workers, per_worker_buffer_size=worker_buffer_size)
        result = result.mp_prefetch(options)
    return result
