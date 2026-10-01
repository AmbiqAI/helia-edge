"""Torch ``DataLoader`` over batches produced by another loader."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch.utils.data


def _plain_arrays(element: Any) -> Any:
    """Copy ndarray subclasses, such as Grain's shared-memory arrays, into plain arrays."""
    if isinstance(element, np.ndarray):
        return element if type(element) is np.ndarray else np.array(element)
    if isinstance(element, Mapping):
        converted = {key: _plain_arrays(value) for key, value in element.items()}
        try:
            return type(element)(converted)
        except TypeError:
            return converted
    if isinstance(element, tuple) and hasattr(element, "_fields"):
        return type(element)(*(_plain_arrays(value) for value in element))
    if isinstance(element, (list, tuple)):
        return type(element)(_plain_arrays(value) for value in element)
    return element


def to_torch_loader(dataset: Iterable[Any]) -> "torch.utils.data.DataLoader":
    """Wrap a re-iterable of NumPy batches, such as a Grain dataset, as a Torch ``DataLoader``.

    The loader adds no batching, shuffling or workers of its own: each ``iter()`` iterates
    ``dataset`` once more in this process and converts its NumPy leaves to tensors, keeping the
    structure (dicts, lists, tuples) and the order. Arrays in worker shared memory are copied
    first, so tensors stay valid after the next element arrives. Do batching and parallel reads
    in ``dataset``.

    Args:
        dataset: Re-iterable whose ``iter()`` starts from the first element.

    Returns:
        torch.utils.data.DataLoader: With ``batch_size=None``, so elements pass through as they are.
    """
    import torch.utils.data

    class _Batches(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(dataset)

    def convert(element: Any) -> Any:
        return torch.utils.data.default_convert(_plain_arrays(element))

    return torch.utils.data.DataLoader(_Batches(), batch_size=None, collate_fn=convert)
