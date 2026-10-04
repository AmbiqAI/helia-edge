"""
# ResNet

## Overview

ResNet is a type of convolutional neural network (CNN) that is commonly used for image classification tasks. ResNet is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow ResNet to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to the original paper [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90).

Parameters are in ``helia_edge.models.resnet_params``.

Functions:
    build: ResNet model from ``ResNetParams``
    generate_bottleneck_block: Generate functional bottleneck block
    generate_residual_block: Generate functional residual block
    resnet_layer: Generate functional ResNet model

## Additions

* Enable 1D and 2D variants.

"""

import keras

from ..layers.convolutional import conv2d
from ..layers.normalization import batch_normalization
from .resnet_params import ResNetParams


def generate_bottleneck_block(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    expansion: int = 4,
    activation: str = "relu6",
) -> keras.Layer:
    """Generate functional bottleneck block.

    Args:
        filters (int): Filter size
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        expansion (int, optional): Expansion factor. Defaults to 4.

    Returns:
        keras.Layer: TF functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        num_chan = x.shape[-1]
        projection = num_chan != filters * expansion or (strides > 1 if isinstance(strides, int) else max(strides) > 1)

        bx = conv2d(filters, 1, 1)(x)
        bx = batch_normalization()(bx)
        bx = keras.layers.Activation(activation)(bx)

        bx = conv2d(filters, kernel_size, strides)(x)
        bx = batch_normalization()(bx)
        bx = keras.layers.Activation(activation)(bx)

        bx = conv2d(filters * expansion, 1, 1)(bx)
        bx = batch_normalization()(bx)

        if projection:
            x = conv2d(filters * expansion, 1, strides)(x)
            x = batch_normalization()(x)
        x = keras.layers.Add()([bx, x])
        x = keras.layers.Activation(activation)(x)
        return x

    return layer


def generate_residual_block(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    activation: str = "relu6",
) -> keras.Layer:
    """Generate functional residual block

    Args:
        filters (int): Filter size
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.

    Returns:
        keras.Layer: TF functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        num_chan = x.shape[-1]
        projection = num_chan != filters or (strides > 1 if isinstance(strides, int) else max(strides) > 1)
        bx = conv2d(filters, kernel_size, strides)(x)
        bx = batch_normalization()(bx)
        bx = keras.layers.Activation(activation)(bx)

        bx = conv2d(filters, kernel_size, 1)(bx)
        bx = batch_normalization()(bx)
        if projection:
            x = conv2d(filters, 1, strides)(x)
            x = batch_normalization()(x)
        x = keras.layers.Add()([bx, x])
        x = keras.layers.Activation(activation)(x)
        return x

    return layer


def resnet_layer(x: keras.KerasTensor, params: ResNetParams) -> keras.KerasTensor:
    """Generate functional ResNet model.
    Args:
        x (keras.KerasTensor): Inputs
        params (ResNetParams): Model parameters.

    Returns:
        keras.KerasTensor: Output tensor
    """

    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x
    # END IF

    if params.input_filters:
        y = conv2d(
            params.input_filters,
            kernel_size=params.input_kernel_size,
            strides=params.input_strides,
        )(y)
        y = batch_normalization()(y)
        y = keras.layers.Activation(params.input_activation)(y)
    # END IF

    for stage, block in enumerate(params.blocks):
        for d in range(block.depth):
            func = generate_bottleneck_block if block.bottleneck else generate_residual_block
            y = func(
                filters=block.filters,
                kernel_size=block.kernel_size,
                strides=block.strides if d == 0 and stage > 0 else 1,
                activation=block.activation,
            )(y)
        # END FOR
    # END FOR

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}_pool")(y)
        if 0 < params.dropout < 1:
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
    params: ResNetParams, input_shape: tuple[int | None, ...], *, batch_size: int | None = None, name: str | None = None
) -> keras.Model:
    """Build a ResNet model.

    Args:
        params (ResNetParams): Model parameters.
        input_shape (tuple[int | None, ...]): Input shape without the batch axis; None for a variable axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``resnet`` unless ``name`` is given.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=resnet_layer(inputs, params), name=name or params.family)
