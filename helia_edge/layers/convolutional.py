"""Convolutional Layers API

This module provides classes to build convolutional layers.

Functions:
    conv1d: 1D convolutional layer using 2D convolutional layer
    conv2d: 2D convolutional layer

Please refer to [Keras Convolutional Layers](https://keras.io/api/layers/convolution_layers/) for additional layers.

"""

import keras


def conv2d(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    padding: str = "same",
    use_bias: bool = False,
    groups: int = 1,
    dilation: int = 1,
    name: str | None = None,
) -> keras.Layer:
    """2D convolutional layer

    Args:
        filters (int): Number of filters
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        groups (int, optional): Number of groups. Defaults to 1.
        dilation (int, optional): Dilation rate. Defaults to 1.
        name (str | None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional 2D conv layer
    """
    name = name + ".conv" if name else None
    return keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        dilation_rate=dilation,
        kernel_initializer="he_normal",
        name=name,
    )


def conv1d(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    name: str | None = None,
) -> keras.Layer:
    """1D convolutional layer using 2D convolutional layer

    Args:
        filters (int): Number of filters
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (int, optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        name (str | None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional 1D conv layer
    """
    name = name + ".conv" if name else None
    return keras.layers.Conv2D(
        filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding=padding,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        name=name,
    )
