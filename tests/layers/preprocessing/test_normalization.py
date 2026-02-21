import numpy as np
import tensorflow as tf

from helia_edge.layers.preprocessing import AugmentationPipeline, Normalization1D, Normalization2D


def test_normalization2d_normalizes_data_and_preserves_labels_dict():
    layer = Normalization2D(mean=[0.5, 0.25, 0.75], variance=[0.25, 1.0, 4.0], epsilon=0.0)
    data = np.array(
        [
            [[[0.5, 0.25, 0.75], [1.0, 0.25, 2.75]]],
            [[[0.0, 1.25, -1.25], [0.5, 0.25, 0.75]]],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 0], dtype=np.int32)

    outputs = layer({"data": tf.constant(data), "labels": tf.constant(labels)}, training=True)

    expected = (data - np.array([0.5, 0.25, 0.75], dtype=np.float32)) / np.sqrt(
        np.array([0.25, 1.0, 4.0], dtype=np.float32)
    )
    np.testing.assert_allclose(np.asarray(outputs["data"]), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(outputs["labels"]), labels)


def test_normalization1d_handles_unbatched_inputs():
    layer = Normalization1D(mean=[1.0, 2.0], variance=[4.0, 9.0], epsilon=0.0)
    data = np.array([[5.0, 8.0], [1.0, 2.0], [3.0, -1.0]], dtype=np.float32)

    outputs = layer(tf.constant(data), training=True)

    expected = (data - np.array([1.0, 2.0], dtype=np.float32)) / np.sqrt(np.array([4.0, 9.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(outputs), expected, rtol=1e-6, atol=1e-6)


def test_normalization2d_inside_augmentation_pipeline():
    pipeline = AugmentationPipeline(
        layers=[
            Normalization2D(mean=[0.0, 0.0, 0.0], variance=[1.0, 4.0, 9.0], epsilon=0.0),
        ]
    )
    data = np.ones((2, 2, 2, 3), dtype=np.float32)
    labels = np.array([2, 3], dtype=np.int32)

    outputs = pipeline({"data": tf.constant(data), "labels": tf.constant(labels)}, training=True)

    expected = data / np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(outputs["data"]), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(outputs["labels"]), labels)
