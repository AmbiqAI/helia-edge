"""
# FLOPs Metrics API

FLOPS calculation for keras.Model or keras.Sequential.

Functions:
    get_flops: Calculate FLOPS for keras.Model or keras.Sequential


"""

import os
import keras


def get_flops(model: keras.Model, batch_size: int | None = None, fpath: os.PathLike | None = None) -> float:
    """Calculate FLOPS for keras.Model or keras.Sequential.
    Ignore operations used in only training mode such as Initialization.

    Note:
        Only tensorflow backend is supported currently.
        Does not support LSTM and GRU.

    Args:
        model (keras.Model|keras.Sequential): Model
        batch_size (int, optional): Batch size. Defaults to None.
        fpath (os.PathLike, optional): Output file path. Defaults to None.

    Returns:
        float: FLOPS

    Example:

    ```python
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    import helia_edge as helia

    model.build()
    flops = helia.metrics.get_flops(model, batch_size=1)
    print(f"FLOPS: {flops/1e6:.2f}M")
    ```
    """
    if keras.backend.backend() != "tensorflow":
        raise ValueError("Only tensorflow backend is supported.")

    import tensorflow as tf

    # pylint: disable=no-name-in-module
    from tensorflow.python.profiler.model_analyzer import profile

    # pylint: disable=no-name-in-module
    from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

    input_signature = [tf.TensorSpec(shape=(batch_size,) + model.input_shape[1:])]
    forward_pass = tf.function(model.call, input_signature=input_signature)
    graph = forward_pass.get_concrete_function().graph
    options = ProfileOptionBuilder.float_operation()
    if fpath:
        options["output"] = f"file:outfile={fpath}"
    graph_info = profile(graph, options=options)
    return float(graph_info.total_float_ops)
