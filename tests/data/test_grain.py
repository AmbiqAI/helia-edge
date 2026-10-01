"""Grain pipelines: order, record IDs and randomness across workers and loaders."""

import itertools
import os
import sys

import numpy as np
import pytest

grain = pytest.importorskip("grain")

from helia_edge.data import to_grain  # noqa: E402


def records(n=10):
    return [{"id": np.int64(i), "x": np.full((4,), i, dtype=np.float32)} for i in range(n)]


def jitter(record, rng):
    return {"id": record["id"], "x": record["x"] + rng.normal(size=record["x"].shape).astype(np.float32)}


def read(dataset):
    iterator = iter(dataset)
    try:
        return [{k: np.array(v, copy=True) for k, v in element.items()} for element in iterator]
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def ids(elements):
    return [int(i) for element in elements for i in np.atleast_1d(element["id"])]


def test_unshuffled_single_epoch_keeps_source_order():
    elements = read(to_grain(records()))
    assert ids(elements) == list(range(10))
    np.testing.assert_array_equal(elements[3]["x"], np.full((4,), 3, dtype=np.float32))


def test_each_epoch_is_a_new_permutation_of_every_record():
    order = ids(read(to_grain(records(), seed=7, shuffle=True, num_epochs=3)))
    epochs = [order[0:10], order[10:20], order[20:30]]
    assert all(sorted(epoch) == list(range(10)) for epoch in epochs)
    assert epochs[0] != epochs[1] and epochs[1] != epochs[2]
    assert epochs[0] != list(range(10))


def test_the_seed_fixes_order_and_augmentation_and_workers_do_not_change_them():
    def run(seed, workers):
        return read(to_grain(records(), seed=seed, shuffle=True, transform=jitter, num_epochs=2, workers=workers))

    a, b, c = run(3, 0), run(3, 2), run(3, 2)
    assert ids(a) == ids(b) == ids(c)
    for x, y, z in zip(a, b, c, strict=True):
        np.testing.assert_array_equal(x["x"], y["x"])
        np.testing.assert_array_equal(y["x"], z["x"])
    other = run(4, 0)
    assert ids(other) != ids(a)
    by_id = {int(e["id"]): e["x"] for e in a[:10]}
    assert any(not np.array_equal(by_id[int(e["id"])], e["x"]) for e in other[:10])


def test_every_pass_replays_the_same_elements():
    dataset = to_grain(records(), seed=9, shuffle=True, transform=jitter, batch_size=4)
    first, second = read(dataset), read(dataset)
    assert ids(first) == ids(second)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a["x"], b["x"])


def test_the_transform_draws_differ_between_epochs():
    elements = read(to_grain(records(), seed=5, transform=jitter, num_epochs=2))
    assert ids(elements) == list(range(10)) * 2
    assert not np.array_equal(elements[0]["x"], elements[10]["x"])


def test_batches_stack_records_and_keep_or_drop_the_remainder():
    kept = read(to_grain(records(), batch_size=4))
    assert [e["x"].shape for e in kept] == [(4, 4), (4, 4), (2, 4)]
    assert ids(kept) == list(range(10))
    dropped = read(to_grain(records(), batch_size=4, drop_remainder=True))
    assert [e["x"].shape for e in dropped] == [(4, 4), (4, 4)]
    across_epochs = read(to_grain(records(), batch_size=4, num_epochs=2))
    assert [e["x"].shape for e in across_epochs] == [(4, 4)] * 5
    assert ids(across_epochs) == list(range(10)) * 2


def tag_process(record, rng):
    return {"id": record["id"], "pid": np.int64(os.getpid())}


def test_workers_run_the_transform_in_other_processes():
    in_process = read(to_grain(records(), seed=0, transform=tag_process))
    assert {int(e["pid"]) for e in in_process} == {os.getpid()}
    in_workers = read(to_grain(records(), seed=0, transform=tag_process, workers=2))
    assert os.getpid() not in {int(e["pid"]) for e in in_workers}
    assert ids(in_workers) == list(range(10))


def test_unbounded_epochs_repeat_until_stopped():
    order = [int(e["id"]) for e in itertools.islice(iter(to_grain(records(3), num_epochs=None)), 7)]
    assert order == [0, 1, 2, 0, 1, 2, 0]


def test_the_torch_loader_yields_the_same_batches_in_the_same_order():
    torch = pytest.importorskip("torch")
    from helia_edge.data import to_torch_loader

    dataset = to_grain(records(), seed=11, shuffle=True, transform=jitter, batch_size=3, workers=2)
    expected = read(dataset)
    for _ in range(2):
        got = list(to_torch_loader(dataset))
        assert len(got) == len(expected)
        for batch, want in zip(got, expected, strict=True):
            assert isinstance(batch["x"], torch.Tensor)
            np.testing.assert_array_equal(batch["id"].numpy(), want["id"])
            np.testing.assert_array_equal(batch["x"].numpy(), want["x"])
    if "keras" not in sys.modules or sys.modules["keras"].backend.backend() == "torch":
        assert "tensorflow" not in sys.modules
