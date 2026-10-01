"""to_torch_loader passes elements of another loader through in order, as tensors."""

from collections import OrderedDict
from typing import NamedTuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helia_edge.data import to_torch_loader  # noqa: E402


class Pair(NamedTuple):
    x: np.ndarray
    y: np.ndarray


class Shared(np.ndarray):
    """Stands in for an array whose memory belongs to a worker."""


def test_structures_and_order_pass_through_on_every_pass():
    batches = [
        {"pair": Pair(np.full((2, 3), i, np.float32), np.array([i, i])), "parts": [np.arange(i + 1)]} for i in range(3)
    ]
    loader = to_torch_loader(batches)
    for _ in range(2):
        got = list(loader)
        assert len(got) == 3
        for batch, want in zip(got, batches, strict=True):
            assert isinstance(batch["pair"], Pair)
            assert isinstance(batch["parts"], list)
            np.testing.assert_array_equal(batch["pair"].x.numpy(), want["pair"].x)
            np.testing.assert_array_equal(batch["pair"].y.numpy(), want["pair"].y)
            np.testing.assert_array_equal(batch["parts"][0].numpy(), want["parts"][0])


def test_ndarray_subclasses_are_copied_into_tensors():
    shared = np.arange(6, dtype=np.float32).reshape(2, 3).view(Shared)
    (batch,) = list(to_torch_loader([{"x": shared}]))
    assert isinstance(batch["x"], torch.Tensor)
    shared[...] = -1
    np.testing.assert_array_equal(batch["x"].numpy(), np.arange(6, dtype=np.float32).reshape(2, 3))


def test_mapping_types_are_kept():
    (batch,) = list(to_torch_loader([OrderedDict(b=np.zeros(2), a=np.ones(2))]))
    assert type(batch) is OrderedDict
    assert list(batch) == ["b", "a"]
