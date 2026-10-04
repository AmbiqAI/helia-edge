"""
# ConvMixer

For more info, refer to the original paper [ConvMixer: Revisiting Convolution in Vision](https://arxiv.org/abs/2201.09792).

Functions:
    build: ConvMixer model from ``ConvMixerParams``
    conv_mixer_block: ConvMixer block
    conv_mixer_layer: ConvMixer layer

"""

from typing import Callable

import keras

from ..layers.normalization import batch_normalization
from .convmixer_params import ConvMixerParams


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


def conv_mixer_layer(x: keras.KerasTensor, params: ConvMixerParams) -> keras.KerasTensor:
    """ConvMixer: https://openreview.net/pdf?id=TVHS5Y4dNvM.

    Args:
        x (keras.KerasTensor): Input tensor
        params (ConvMixerParams): Model parameters.

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
        if params.num_classes is not None:
            y = keras.layers.Dense(params.num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    return y


def build(
    params: ConvMixerParams,
    input_shape: tuple[int | None, ...],
    *,
    batch_size: int | None = None,
    name: str | None = None,
) -> keras.Model:
    """Build a ConvMixer model.

    Args:
        params (ConvMixerParams): Model parameters.
        input_shape (tuple[int | None, ...]): Input shape without the batch axis; None for a variable axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``convmixer`` unless ``name`` is given.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=conv_mixer_layer(inputs, params), name=name or params.family)
