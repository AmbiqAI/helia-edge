"""
# TsMixer Model

## Overview

TsMixer is a fully MLP-based architecture for time series data.

For more info, refer to the original paper [TsMixer: An All-MLP Architecture for Time Series](https://arxiv.org/abs/2303.06053).

Classes:
    TsMixerParams: TsMixer parameters
    TsMixerModel: Helper class to generate model from parameters

Functions:
    ts_block: Residual block of TsMixer
    norm_layer: Normalization layer
    tsmixer_layer: TsMixer layer

"""

from typing import Literal

import keras
from pydantic import BaseModel, Field


class TsMixerBlockParams(BaseModel):
    """TsMixer block parameters

    Attributes:
        norm (Literal["batch", "layer"]): Normalization type
        activation (Literal["relu", "gelu"]): Activation type
        dropout (float): Dropout rate
        ff_dim (int): Feed forward dimension
    """

    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    activation: Literal["relu", "gelu"] | None = Field(default="relu", description="Activation type")
    dropout: float | None = Field(default=None, description="Dropout rate")
    ff_dim: int | None = Field(default=None, description="Feed forward dimension")


class TsMixerParams(BaseModel):
    """TsMixer parameters

    Attributes:
        blocks (list[TsBlockParams]): TsMixer blocks
        name (str): Model name
    """

    blocks: list[TsMixerBlockParams] = Field(default_factory=list, description="UNext blocks")
    name: str = Field(default="TsMixer", description="Model name")


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

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Temporal Linear
        y = norm_layer(params.norm, name=f"{name}_TL")(x)

        y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Channel, Input Length]
        y = keras.layers.Dense(y.shape[-1], activation=params.activation, name=f"{name}_TL_DENSE")(y)
        y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Input Length, Channel]
        y = keras.layers.Dropout(params.dropout, name=f"{name}_TL_DROP")(y)
        res = y + x

        # Feature Linear
        y = norm_layer(params.norm, name=f"{name}_FL")(res)
        y = keras.layers.Dense(params.ff_dim, activation=params.activation, name=f"{name}_FL_DENSE")(
            y
        )  # [Batch, Input Length, FF_Dim]
        y = keras.layers.Dropout(params.dropout, name=f"{name}_FL_DROP")(y)

        y = keras.layers.Dense(x.shape[-1], name=f"{name}_RL_DENSE")(y)  # [Batch, Input Length, Channel]
        y = keras.layers.Dropout(params.dropout, name=f"{name}_RL_DROP")(y)
        return y + res

    return layer


def tsmixer_layer(inputs: keras.KerasTensor, params: any, num_classes: int) -> keras.KerasTensor:
    """TsMixer layer

    Args:
        inputs (keras.KerasTensor): Input tensor
        params (any): Model parameters
        num_classes (int): Number of classes

    Returns:
        keras.KerasTensor: Output tensor
    """
    y = inputs
    for block in range(params.blocks):
        y = ts_block(block)(y)

    # if target_slice:
    #     y = y[:, :, target_slice]

    y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Channel, Input Length]
    y = keras.layers.Dense(num_classes)(y)  # [Batch, Channel, Output Length]
    y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Output Length, Channel])

    return y


class TsMixerModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: TsMixerParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = TsMixerParams(**params)
        return tsmixer_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: TsMixerParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = TsMixerModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
