"""
# ResNet

## Overview

ResNet is a type of convolutional neural network (CNN) that is commonly used for image classification tasks. ResNet is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow ResNet to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to the original paper [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90).

Classes:
    ResNetParams: ResNet parameters
    ResNetModel: Helper class to generate model from parameters

Functions:
    generate_bottleneck_block: Generate functional bottleneck block
    generate_residual_block: Generate functional residual block
    resnet_layer: Generate functional ResNet model

## Additions

* Enable 1D and 2D variants.

"""

import keras
from pydantic import BaseModel, Field

from ..layers.normalization import batch_normalization
from ..layers.convolutional import conv2d


class ResNetBlockParams(BaseModel):
    """ResNet block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        bottleneck (bool): Use bottleneck blocks
        activation (str): Activation function

    """

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    bottleneck: bool = Field(default=False, description="Use bottleneck blocks")
    activation: str = Field(default="relu6", description="Activation function")


class ResNetParams(BaseModel):
    """ResNet parameters

    Attributes:
        blocks (list[ResNetBlockParams]): ResNet blocks
        input_filters (int): Input filters
        input_kernel_size (int | tuple[int, int]): Input kernel size
        input_strides (int | tuple[int, int]): Input stride
        input_activation (str): Input activation
        include_top (bool): Include top
        output_activation (str | None): Output activation
        dropout (float): Dropout rate
        name (str): Model name

    """

    blocks: list[ResNetBlockParams] = Field(default_factory=list, description="ResNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_activation: str = Field(default="relu6", description="Input activation")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    dropout: float = Field(default=0.2, description="Dropout rate")
    name: str = Field(default="ResNet", description="Model name")


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
        projection = num_chan != filters * expansion or (strides > 1 if isinstance(strides, int) else strides[0] > 1)

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
        projection = num_chan != filters or (strides > 1 if isinstance(strides, int) else strides[0] > 1)
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


def resnet_layer(
    x: keras.KerasTensor,
    params: ResNetParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """Generate functional ResNet model.
    Args:
        x (keras.KerasTensor): Inputs
        params (ResNetParams): Model parameters.
        num_classes (int, optional): Number of class outputs. Defaults to None.

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

        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)

        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    # Only reshape if needed
    elif requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


class ResNetModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: ResNetParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = ResNetParams(**params)
        return resnet_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: ResNetParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = ResNetModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
