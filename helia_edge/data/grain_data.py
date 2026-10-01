"""Grain pipelines over a ``DataSource``."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .source import DataSource

if TYPE_CHECKING:
    import grain


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
    ``seed``, not on ``workers``.

    Args:
        source: Records addressed by index.
        seed: Seed for the shuffle and for the per-record generator passed to ``transform``.
            Required when ``shuffle`` is set or ``transform`` is given.
        shuffle: Shuffle the records of every epoch.
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
        ValueError: If ``seed`` is missing when needed, or a count is out of range.
        ImportError: If Grain is not installed (``helia-edge[grain]``).
    """
    if seed is None and (shuffle or transform is not None):
        raise ValueError("to_grain needs a seed when shuffle is set or a transform is given")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if num_epochs is not None and num_epochs < 1:
        raise ValueError(f"num_epochs must be positive or None; got {num_epochs}")
    if workers < 0 or worker_buffer_size < 1:
        raise ValueError(f"workers must be >= 0 and worker_buffer_size >= 1; got {workers}, {worker_buffer_size}")
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
