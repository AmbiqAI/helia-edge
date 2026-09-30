"""Optional Grain CPU example; set KERAS_BACKEND=torch before importing Keras.

Install Grain separately. The pipeline owns per-record randomness and IPC output
lifetimes; EDGE owns the same transforms used outside Grain.
"""

import numpy as np


def preprocess_record(record, rng):
    """Apply existing layers with a Grain-owned per-record seed on the CPU."""
    import keras

    from helia_edge.layers.preprocessing import FirFilter, Normalization1D, RandomGaussianNoise1D

    with keras.device("cpu"):
        output = Normalization1D(mean=0.0, variance=4.0)(record)
        output = FirFilter(b=np.array([0.25, 0.5, 0.25], dtype=np.float32))(output)
        output = RandomGaussianNoise1D(factor=(0.1, 0.1), seed=int(rng.integers(0, 2**31 - 1)))(output, training=True)
        return keras.tree.map_structure(keras.ops.convert_to_numpy, output)


def dataset(records, *, seed=123, workers=0):
    """Build an optional Grain iterator without importing TensorFlow."""
    import grain

    result = grain.MapDataset.source(records).random_map(preprocess_record, seed=seed).to_iter_dataset()
    if workers:
        result = result.mp_prefetch(grain.MultiprocessingOptions(num_workers=workers, per_worker_buffer_size=1))
    return result


def owned_records(records, *, seed=123, workers=0):
    """Copy IPC arrays explicitly and close the iterator before returning."""
    import keras

    iterator = iter(dataset(records, seed=seed, workers=workers))
    try:
        return [keras.tree.map_structure(lambda x: np.array(x, copy=True), row) for row in iterator]
    finally:
        iterator.close()
