"""
# RegNet Model API

This module provides utility functions to generate RegNet models.

Fore more information, please refer to the following paper: https://arxiv.org/abs/2101.00590

Parameters are in ``helia_edge.models.regnet_params``.

Functions:
    build: RegNet model from ``RegNetParams``
    regnet_core: RegNet core
    regnet_layer: Generate RegNet model

"""

from typing import Callable, Literal

import keras

from ..layers.convolutional import conv2d
from ..layers.normalization import batch_normalization
from ..layers.squeeze_excite import se_layer
from .regnet_params import RegNetBlockParam, RegNetParams
from .utils import make_divisible


def yblock(
    output_filters: int = 0,
    group_width: int = 0,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 4,
    activation: str = "relu6",
    name: str | None = None,
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """RegNet Y-Block

    Args:
        output_filters (int, optional): Number of output filters. Defaults to 0.
        group_width (int, optional): Group width. Defaults to 0.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 4.
        activation (str, optional): Activation function. Defaults to "relu6".
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[keras.KerasTensor], keras.KerasTensor]: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1]
        groups = output_filters // group_width
        use_skip = (input_filters != output_filters) or (
            strides != 1 if isinstance(strides, int) else any(s != 1 for s in strides)
        )

        name_ex = f"{name}_exp" if name else None
        y = conv2d(output_filters, kernel_size=(1, 1), name=name_ex)(x)
        y = batch_normalization(name=name_ex)(y)
        y = keras.layers.Activation(activation, name=f"{name_ex}_act")(y)

        y = keras.layers.Conv2D(
            output_filters,
            kernel_size=kernel_size,
            strides=strides,
            groups=groups,
            padding="same",
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(),  # type: ignore
        )(y)
        y = batch_normalization()(y)
        y = keras.layers.Activation(activation)(y)

        if se_ratio:
            y = se_layer(ratio=se_ratio)(y)

        y = conv2d(output_filters, kernel_size=(1, 1), strides=(1, 1))(y)
        y = batch_normalization()(y)
        y = keras.layers.Activation(activation)(y)

        if use_skip:
            x = conv2d(output_filters, kernel_size=(1, 1), strides=strides)(x)
            x = batch_normalization()(x)

        y = keras.layers.Add()([y, x])
        y = keras.layers.Activation(activation)(y)
        return y

    return layer


def zblock(
    output_filters: int = 0,
    group_width: int = 0,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 4,
    activation: str = "relu6",
    name: str | None = None,
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """RegNet X-Block

    Args:
        output_filters (int, optional): Number of output filters. Defaults to 0.
        group_width (int, optional): Group width. Defaults to 0.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 4.
        activation (str, optional): Activation function. Defaults to "relu6".
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[keras.KerasTensor], keras.KerasTensor]: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1]
        groups = input_filters // group_width
        use_add = input_filters == output_filters and (
            strides == 1 if isinstance(strides, int) else all(s == 1 for s in strides)
        )
        expand_ratio = 2

        name_ex = f"{name}_exp" if name else None
        filters = input_filters * expand_ratio
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), name=name_ex)(x)
        y = batch_normalization(name=name_ex)(y)
        y = keras.layers.Activation(activation, name=f"{name_ex}_act")(y)

        y = keras.layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            groups=groups,
            padding="same",
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(),  # type: ignore
        )(y)
        y = batch_normalization()(y)
        y = keras.layers.Activation(activation)(y)

        if se_ratio:
            y = se_layer(ratio=se_ratio)(y)

        y = conv2d(output_filters, kernel_size=(1, 1), strides=(1, 1))(y)
        y = batch_normalization()(y)

        if use_add:
            y = keras.layers.Add()([y, x])

        return y

    return layer


def regnet_core(
    blocks: list[RegNetBlockParam],
    block_style: Literal["y", "z"] = "y",
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """RegNet core

    Args:
        blocks (list[RegNetBlockParam]): Block params
        block_style (float, optional): Block style. Defaults to 'y'.

    Returns:
        Callable[[keras.KerasTensor], keras.KerasTensor]: Core
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        block_fn = yblock if block_style == "y" else zblock
        for i, block in enumerate(blocks):
            filters = make_divisible(block.filters, 8)
            for d in range(block.depth):
                name = f"stage{i}_block{d + 1}"
                x = block_fn(
                    output_filters=filters,
                    group_width=block.group_width,
                    kernel_size=block.kernel_size,
                    strides=block.strides,
                    se_ratio=block.se_ratio,
                    activation=block.activation,
                    name=name,
                )(x)
            # END FOR
        # END FOR
        return x

    return layer


def regnet_layer(x: keras.KerasTensor, params: RegNetParams) -> keras.KerasTensor:
    """Create RegNet TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (RegNetParams): Model parameters.

    Returns:
        keras.KerasTensor: Output tensor
    """
    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x
    # END IF

    # Stem
    if params.input_filters > 0:
        name = "stem"
        filters = make_divisible(params.input_filters, 8)
        y = conv2d(filters, kernel_size=(3, 3), strides=params.input_strides, name=name)(y)
        y = batch_normalization(name=name)(y)
        y = keras.layers.Activation(params.input_activation, name=f"{name}_act")(y)
    else:
        y = x

    y = regnet_core(blocks=params.blocks, block_style=params.block_style)(y)

    if params.output_filters:
        name = "neck"
        filters = make_divisible(params.output_filters, 8)
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=name)(y)
        y = batch_normalization(name=name)(y)
        y = keras.layers.Activation(params.output_activation, name=f"{name}_act")(y)

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}_pool")(y)

        if params.dropout > 0 and params.dropout < 1:
            y = keras.layers.Dropout(params.dropout)(y)
        if params.num_classes is not None:
            y = keras.layers.Dense(params.num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    # Only reshape if needed
    elif requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


def build(
    params: RegNetParams, input_shape: tuple[int, ...], *, batch_size: int | None = None, name: str | None = None
) -> keras.Model:
    """Build a RegNet model.

    Args:
        params (RegNetParams): Model parameters.
        input_shape (tuple[int, ...]): Input shape without the batch axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``regnet`` unless ``name`` is given.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=regnet_layer(inputs, params), name=name or params.family)
