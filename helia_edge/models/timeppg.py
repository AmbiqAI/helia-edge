# Copyright (C) 2021 Politecnico di Torino, Italy
# Licensed under Apache-2.0; see licenses/timeppg-apache-2.0.txt.
"""TimePPG heart-rate regressor from PPG and accelerometer windows.

Adapted from eml-eda/q-ppg (revision ddf3866d,
precision_search/model/TimePPG_float.py). Inputs are channels-last
(time, channels) windows; the reference uses 256 samples of BVP at 32 Hz plus
three accelerometer axes. Constructors initialize weights only: upstream
publishes no trained checkpoints, so these models are untrained.
"""

import keras

from .timeppg_params import TimePPGParams

# (dilation, receptive field) per temporal block, upstream order.
_TEMPORAL = ((2, 5), (2, 5), (4, 9), (4, 9), (8, 17), (8, 17))
# (kernel, stride, padding) per strided convolution block.
_CONV = ((5, 1, 2), (5, 2, 2), (5, 4, 4))


def _norm_act(x, name):
    # PyTorch BatchNorm1d defaults: eps 1e-5, momentum 0.1.
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f"{name}_bn")(x)
    return keras.layers.Activation("relu6", name=f"{name}_relu6")(x)


def _temporal_block(x, filters, dilation, receptive_field, name):
    kernel = -(-receptive_field // dilation)
    x = keras.layers.Conv1D(
        filters, kernel, dilation_rate=dilation, padding="same", use_bias=False, name=f"{name}_conv"
    )(x)
    return _norm_act(x, name)


def _conv_block(x, filters, kernel, stride, padding, name):
    x = keras.layers.ZeroPadding1D(padding, name=f"{name}_pad")(x)
    x = keras.layers.Conv1D(filters, kernel, strides=stride, use_bias=False, name=f"{name}_conv")(x)
    x = keras.layers.AveragePooling1D(2, name=f"{name}_pool")(x)
    return _norm_act(x, name)


def build(
    params: TimePPGParams,
    input_shape: tuple[int | None, ...],
    *,
    batch_size: int | None = None,
    name: str | None = None,
) -> keras.Model:
    """Construct an untrained TimePPG regressor with one linear output.

    The input is ``(time, channels)`` with a known time length that survives the 64x downsampling. The
    flattened features are channel-major, matching the upstream PyTorch layout.

    Args:
        params (TimePPGParams): Model parameters.
        input_shape (tuple[int | None, ...]): ``(time, channels)``, both known, time at least 64.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``timeppg`` unless ``name`` is given.
    """
    if len(input_shape) != 2:
        raise ValueError("input_shape must be (time, channels)")
    if input_shape[0] is None or input_shape[0] < 64 or input_shape[1] is None:
        raise ValueError("inputs need a known time length of at least 64 and a known channel count")
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")

    ch = params.channels
    x = inputs
    stages = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    for stage, (t0, t1, cb) in enumerate(stages):
        for index in (t0, t1):
            dilation, receptive_field = _TEMPORAL[index - stage]
            x = _temporal_block(x, ch[index], dilation, receptive_field, name=f"tcb{stage}{index - t0}")
        kernel, stride, padding = _CONV[stage]
        x = _conv_block(x, ch[cb], kernel, stride, padding, name=f"cb{stage}")
    x = keras.layers.Permute((2, 1), name="channel_major")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    for index, width in enumerate(ch[9:]):
        x = keras.layers.Dense(width, use_bias=False, name=f"regr{index}_dense")(x)
        x = _norm_act(x, f"regr{index}")
    outputs = keras.layers.Dense(1, name="out_neuron")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name or params.family)
