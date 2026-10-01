"""to_tf_dataset passes elements of another loader through in order."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from helia_edge.data import to_grain, to_tf_dataset  # noqa: E402


def test_elements_pass_through_in_order_on_every_pass():
    batches = [(np.full((2, 3), i, np.float32), np.array([i, i], np.int32)) for i in range(3)]
    signature = (tf.TensorSpec((None, 3), tf.float32), tf.TensorSpec((None,), tf.int32))
    dataset = to_tf_dataset(batches, signature)
    for _ in range(2):
        got = list(dataset.as_numpy_iterator())
        assert len(got) == 3
        for (x, y), (ex, ey) in zip(got, batches, strict=True):
            np.testing.assert_array_equal(x, ex)
            np.testing.assert_array_equal(y, ey)


def test_grain_batches_reach_tf_data_unchanged():
    pytest.importorskip("grain")
    source = [{"id": np.int64(i), "x": np.full((4,), i, np.float32)} for i in range(10)]
    dataset = to_grain(source, seed=2, shuffle=True, batch_size=4, num_epochs=2, workers=2)
    signature = {"id": tf.TensorSpec((None,), tf.int64), "x": tf.TensorSpec((None, 4), tf.float32)}
    expected = [{k: np.array(v, copy=True) for k, v in e.items()} for e in dataset]
    got = list(to_tf_dataset(dataset, signature).as_numpy_iterator())
    assert [list(e["id"]) for e in got] == [list(e["id"]) for e in expected]
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g["x"], e["x"])
