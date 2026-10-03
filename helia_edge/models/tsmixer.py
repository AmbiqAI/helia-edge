"""
# TsMixer Model

## Overview

TsMixer is a fully MLP-based architecture for time series data.

For more info, refer to the original paper [TsMixer: An All-MLP Architecture for Time Series](https://arxiv.org/abs/2303.06053).

Parameters are in ``helia_edge.models.tsmixer_params``.

Functions:
    build: TsMixer model from ``TsMixerParams``
    ts_block: Residual block of TsMixer
    norm_layer: Normalization layer
    tsmixer_layer: TsMixer layer

"""

import keras

from .tsmixer_params import TsMixerBlockParams, TsMixerParams


def norm_layer(norm: str, name: str) -> keras.Layer:
    """Normalization layer

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """Functional normalization layer

        Args:
            x (keras.KerasTensor): Input tensor

        Returns:
            keras.KerasTensor: Output tensor
        """
        if norm == "batch":
            return keras.layers.BatchNormalization(axis=[-2, -1], name=f"{name}_BN")(x)
        if norm == "layer":
            return keras.layers.LayerNormalization(axis=[-2, -1], name=f"{name}_LN")(x)
        return x

    return layer


def ts_block(params: TsMixerBlockParams, name: str) -> keras.Layer:
    """Residual block of TSMixer.

    Args:
        params (TsBlockParams): Block parameters
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def dropout(y: keras.KerasTensor, suffix: str) -> keras.KerasTensor:
        return keras.layers.Dropout(params.dropout, name=f"{name}_{suffix}")(y) if params.dropout else y

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Temporal Linear
        y = norm_layer(params.norm, name=f"{name}_TL")(x)

        y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Channel, Input Length]
        y = keras.layers.Dense(y.shape[-1], activation=params.activation, name=f"{name}_TL_DENSE")(y)
        y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Input Length, Channel]
        y = dropout(y, "TL_DROP")
        res = y + x

        # Feature Linear
        y = norm_layer(params.norm, name=f"{name}_FL")(res)
        ff_dim = params.ff_dim or x.shape[-1]
        y = keras.layers.Dense(ff_dim, activation=params.activation, name=f"{name}_FL_DENSE")(
            y
        )  # [Batch, Input Length, FF_Dim]
        y = dropout(y, "FL_DROP")

        y = keras.layers.Dense(x.shape[-1], name=f"{name}_RL_DENSE")(y)  # [Batch, Input Length, Channel]
        y = dropout(y, "RL_DROP")
        return y + res

    return layer


def tsmixer_layer(inputs: keras.KerasTensor, params: any) -> keras.KerasTensor:
    """TsMixer layer

    Args:
        inputs (keras.KerasTensor): Input tensor
        params (any): Model parameters

    Returns:
        keras.KerasTensor: Output tensor
    """
    y = inputs
    for i, block in enumerate(params.blocks):
        y = ts_block(block, name=f"B{i + 1}")(y)

    # if target_slice:
    #     y = y[:, :, target_slice]

    y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Channel, Input Length]
    y = keras.layers.Dense(params.num_classes)(y)  # [Batch, Channel, Output Length]
    y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Output Length, Channel])

    return y


def build(
    params: TsMixerParams, input_shape: tuple[int, ...], *, batch_size: int | None = None, name: str | None = None
) -> keras.Model:
    """Build a TsMixer model.

    Args:
        params (TsMixerParams): Model parameters.
        input_shape (tuple[int, ...]): Input shape without the batch axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``tsmixer`` unless ``name`` is given.
    """
    if params.num_classes is None:
        raise ValueError("TsMixer needs num_classes")
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=tsmixer_layer(inputs, params), name=name or params.family)
