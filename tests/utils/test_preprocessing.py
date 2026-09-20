"""Coverage, ordering and sampling contracts for generator adapters."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from helia_edge.utils import create_interleaved_dataset_from_generator as make_dataset


@pytest.fixture
def schedule():
    return json.loads((Path(__file__).parents[1] / "fixtures/sleepkit-finite-schedule.json").read_text())


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("mode,ordered", [("global", True), ("finite", True), ("finite", False)])
def test_finite_schedule_preserves_examples_labels_and_clocks(schedule, workers, mode, ordered):
    subjects = {subject["id"]: subject["eligible_contexts"] for subject in schedule["subjects"]}
    ids = schedule["example_ids"]

    def samples(subject_ids):
        for subject in subject_ids:
            for context in range(subjects[subject]):
                logical_id = f"{subject}/c{context}"
                index = ids.index(logical_id)
                yield logical_id, np.int32(index % 2), np.float64(index * 2.5)

    ds = make_dataset(
        samples,
        iter,
        list(subjects),
        (tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.float64)),
        num_workers=workers,
        stream_mode=mode,
        deterministic=ordered,
    ).batch(schedule["batch_size"])
    for _ in range(2):
        batches = list(ds.as_numpy_iterator())
        assert len(batches) == 4 and len(batches[-1][0]) == 1
        rows = [(name.decode(), int(label), float(clock)) for batch in batches for name, label, clock in zip(*batch)]
        expected = [(name, i % 2, i * 2.5) for i, name in enumerate(ids)]
        assert sorted(rows) == sorted(expected)
        if ordered:
            assert rows == expected


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_repeated_global_schedule_is_worker_independent(workers):
    # Caller deliberately weights ID 0 twice; partitioning must not reinterpret it.
    ids = [0, 0, 1, 2, 3]

    def repeated(subjects):
        yield from itertools.cycle(subjects)

    def samples(subjects):
        for subject in subjects:
            for window in range(subject + 1):
                yield np.int32(subject), np.int32(window)

    ds = make_dataset(samples, repeated, ids, (tf.TensorSpec((), tf.int32),) * 2, num_workers=workers)
    actual = list(ds.take(55).as_numpy_iterator())
    expected = list(itertools.islice(samples(repeated(ids)), 55))
    assert actual == expected


@pytest.mark.parametrize("mode", ["global", "finite"])
def test_empty_ids_do_not_invoke_callbacks(mode):
    def forbidden(*args):
        raise AssertionError("Empty streams must not initialize readers")

    ds = make_dataset(forbidden, forbidden, [], tf.TensorSpec((2,), tf.float32), stream_mode=mode)
    assert ds.element_spec == tf.TensorSpec((2,), tf.float32)
    assert list(ds.as_numpy_iterator()) == []


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_workers(workers):
    with pytest.raises(ValueError, match="positive integer"):
        make_dataset(iter, iter, [], tf.TensorSpec((), tf.int32), num_workers=workers)


def test_invalid_mode():
    with pytest.raises(ValueError, match="stream_mode"):
        make_dataset(iter, iter, [], tf.TensorSpec((), tf.int32), stream_mode="repeated")


def test_optional_preprocess_and_input_mutation():
    ids = [1, 2, 3]
    ds = make_dataset(iter, iter, ids, tf.TensorSpec((), tf.int32), preprocess=lambda x: x * 2)
    ids.clear()
    assert list(ds.as_numpy_iterator()) == [2, 4, 6]


def test_mutating_id_generator_gets_fresh_ids_each_iteration():
    def destructive(ids):
        while ids:
            yield ids.pop()

    ds = make_dataset(iter, destructive, [1, 2, 3], tf.TensorSpec((), tf.int32))
    assert list(ds.as_numpy_iterator()) == [3, 2, 1]
    assert list(ds.as_numpy_iterator()) == [3, 2, 1]
