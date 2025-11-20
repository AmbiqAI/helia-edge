"""
# ConvMixer

For more info, refer to the original paper [ConvMixer: Revisiting Convolution in Vision](https://arxiv.org/abs/2201.09792).

Classes:
    ConvMixerParams: ConvMixer parameters
    ConvMixerModel: Helper class to generate model from parameters

Functions:
    conv_mixer_block: ConvMixer block
    conv_mixer_layer: ConvMixer layer

"""

from typing import Callable

import keras
from pydantic import BaseModel, Field

from ..layers.normalization import batch_normalization


class ConvMixerParams(BaseModel):
    """ConvMixer parameters

    Attributes:
        filters (int): Number of filters per layer
        depth (int): Network depth
        kernel_size (int): Filter size
        patch_size (int): Patch size
        include_top (bool): Include top
        output_activation (str | None): Output activation
        name (str): Model name

    """

    filters: int = Field(default=256, description="# filters per layer")
    depth: int = Field(default=8, description="Network depth")
    kernel_size: int = Field(default=5, description="Filter size")
    patch_size: int = Field(default=2, description="Patch size")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="RegNet", description="Model name")


def conv_mixer_block(filters: int, kernel_size: int) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ConvMixer block

    Args:
        filters (int): Number of filters
        kernel_size (int): Kernel size
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Depthwise convolution.
        x0 = x
        x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
        x = keras.layers.Activation("gelu")(x)
        x = batch_normalization()(x)
        # Residual
        x = keras.layers.Add()([x, x0])

        # Pointwise convolution.
        x = keras.layers.Conv2D(filters, kernel_size=1)(x)
        x = keras.layers.Activation("gelu")(x)
        x = batch_normalization()(x)
        return x

    return layer


def conv_mixer_layer(
    x: keras.KerasTensor,
    params: ConvMixerParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """ConvMixer: https://openreview.net/pdf?id=TVHS5Y4dNvM.

    Args:
        x (keras.KerasTensor): Input tensor
        params (ConvMixerParams): Model parameters.
        num_classes (int, optional): Number of classes.

    Returns:
        keras.KerasTensor: Model output
    """
    # Extract patch embeddings
    y = keras.layers.Conv2D(
        filters=params.filters,
        kernel_size=params.patch_size,
        strides=params.patch_size,
        padding="same",
        use_bias=True,
    )(x)
    y = keras.layers.Activation("gelu")(y)
    y = batch_normalization()(y)

    # ConvMixer blocks
    for _ in range(params.depth):
        y = conv_mixer_block(filters=params.filters, kernel_size=params.kernel_size)(y)

    # Classification block
    if params.include_top:
        y = keras.layers.GlobalAvgPool2D(keepdims=False)(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    return y


class ConvMixerModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: ConvMixerParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = ConvMixerParams(**params)
        return conv_mixer_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: ConvMixerParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = ConvMixerModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
