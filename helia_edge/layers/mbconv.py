"""
# MBConv Layer API

This module provides classes to build mobile inverted bottleneck convolutional layers.

Classes:
    MBConvParams: MBConv parameters

Functions:
    mbconv_block: MBConv block w/ expansion and SE

"""

from typing import Callable
from collections.abc import Iterable

import keras
from pydantic import BaseModel, Field

from .squeeze_excite import se_layer
from .convolutional import conv2d
from .normalization import batch_normalization


class MBConvParams(BaseModel):
    """MBConv parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        ex_ratio (float): Expansion ratio
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        se_ratio (float): Squeeze Excite ratio
        droprate (float): Drop rate
        bn_momentum (float): Batch normalization momentum
        activation (str): Activation function
    """

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")
    bn_momentum: float = Field(default=0.9, description="Batch normalization momentum")
    activation: str = Field(default="relu6", description="Activation function")


def mbconv_block(
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 8,
    droprate: float = 0,
    bn_momentum: float = 0.9,
    activation: str | Callable = "relu6",
    name: str | None = None,
) -> keras.Layer:
    """MBConv block w/ expansion and SE

    This layer can support 1D inputs by providing a dummy dimension.
    In such cases, the kernel_size and strides should be adjusted accordingly.

    Args:
        output_filters (int): Number of output filter channels
        expand_ratio (float, optional): Expansion ratio. Defaults to 1.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 8.
        droprate (float, optional): Drop rate. Defaults to 0.
        bn_momentum (float, optional): Batch normalization momentum. Defaults to 0.9.
        activation (str, optional): Activation function. Defaults to "relu6".
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        keras.Layer: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1] if keras.backend.image_data_format() == "channels_last" else x.shape[1]
        stride_len = strides if isinstance(strides, int) else sum(strides) / len(strides)
        is_symmetric = isinstance(kernel_size, Iterable) and kernel_size[0] == kernel_size[1]
        is_downsample = not is_symmetric and stride_len > 1

        add_residual = input_filters == output_filters and stride_len == 1
        # Expand: narrow -> wide
        if expand_ratio != 1:
            name_ex = f"{name}_exp" if name else None
            name_ex_bn = f"{name}_exp_bn" if name else None
            name_ex_act = f"{name}_exp_act" if name else None
            filters = int(input_filters * expand_ratio)
            y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), name=name_ex)(x)
            y = batch_normalization(name=name_ex_bn)(y)
            y = keras.layers.Activation(activation, name=name_ex_act)(y)
        else:
            y = x

        # Apply: wide -> wide
        name_dp = f"{name}_dp" if name else None
        name_dp_bn = f"{name}_dp_bn" if name else None
        name_dp_act = f"{name}_dp_act" if name else None
        y = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides if is_symmetric else (1, 1),
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            name=name_dp,
        )(y)
        y = batch_normalization(name=name_dp_bn, momentum=bn_momentum)(y)
        y = keras.layers.Activation(activation, name=name_dp_act)(y)
        # NOTE: DepthwiseConv2D only supports equal size stride -> use maxpooling as needed
        if is_downsample:
            y = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y)
        # END IF

        # SE: wide -> wide
        if se_ratio:
            name_se = f"{name}_se" if name else None
            y = se_layer(ratio=se_ratio * expand_ratio, name=name_se)(y)

        # Reduce: wide -> narrow
        name_rd = f"{name}_red" if name else None
        name_rd_bn = f"{name}_red_bn" if name else None
        y = conv2d(
            output_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name=name_rd,
        )(y)
        y = batch_normalization(name=name_rd_bn, momentum=bn_momentum)(y)

        # No activation

        if add_residual:
            name_res = f"{name}_res" if name else None
            name_res_dp = f"{name}_res_dp" if name else None
            if droprate > 0:
                y = keras.layers.Dropout(droprate, noise_shape=(None, 1, 1, 1), name=name_res_dp)(y)
            y = keras.layers.add([x, y], name=name_res)
        return y

    # END DEF

    return layer
