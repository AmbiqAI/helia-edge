"""
# MobileOne

## Overview

MobileOne is a fully convolutional neural network designed to have minimal latency when running in mobile/edge devices. The architecture consists of a series of depthwise separable convolutions and squeeze and excitation (SE) blocks. The network also uses standard batch normalization and ReLU activations that can be easily fused into the convolutional layers. Lastly, the network uses over-parameterized branches to improve training, yet can be merged into a single branch during inference.

For more info, refer to the original paper [MobileOne: An Improved One millisecond Mobile Backbone](https://doi.org/10.48550/arXiv.2206.04040).

Classes:
    MobileOneParams: MobileOne parameters
    MobileOneBlockParams: MobileOne block parameters
    MobileOneModel: Helper class to generate model from parameters

Functions:
    mobileone_block: MobileOne block
    mobileone_layer: MobileOne layer

---

## Additions

The MobileOne architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Enable dilated convolutions.

## Usage

```python
import keras
from helia_edge.models import MobileOne, MobileOneParams, MobileOneBlockParams

inputs = keras.Input(shape=(800, 1), name="inputs")

model = MobileOne.model_from_params(
    x=inputs,
    params=MobileOneParams(
        input_filters=24,
        input_kernel_size=(1, 7),
        input_strides=(1, 2),
        blocks=[
            MobileOneBlockParams(filters=32, depth=2, kernel_size=(1, 7), strides=(1, 2), se_ratio=2, se_depth=2, num_conv_branches=2)
        ],
        include_top=True,
        model_name="MobileOne",
    ),
)

model.summary()

```
"""

import keras
from pydantic import BaseModel, Field

from ..layers import se_layer
from ..layers.normalization import batch_normalization
from ..layers.convolutional import conv2d


class MobileOneBlockParams(BaseModel):
    """MobileOne block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        padding (int | tuple[int, int]): Padding size
        se_ratio (float): Squeeze Excite ratio
        se_depth (int): Depth length to apply SE
        num_conv_branches (int): Number of conv branches
        activation (str): Activation function

    """

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    padding: int | tuple[int, int] = Field(default=0, description="Padding size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    se_depth: int = Field(default=0, description="Depth length to apply SE")
    num_conv_branches: int = Field(default=2, description="# conv branches")
    activation: str = Field(default="relu6", description="Activation function")


class MobileOneParams(BaseModel):
    """MobileOne parameters

    Attributes:
        blocks (list[MobileOneBlockParams]): MobileOne blocks
        input_filters (int): Input filters
        input_kernel_size (int | tuple[int, int]): Input kernel size
        input_strides (int | tuple[int, int]): Input stride
        input_padding (int | tuple[int, int]): Input padding
        include_top (bool): Include top
        output_activation (str | None): Output activation
        dropout (float): Dropout rate
        name (str): Model name

    """

    blocks: list[MobileOneBlockParams] = Field(default_factory=list, description="MobileOne blocks")

    input_filters: int = Field(default=3, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_padding: int | tuple[int, int] = Field(default=1, description="Input padding")

    # output_filters: int = Field(default=0, description="Output filters")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    dropout: float = Field(default=0.2, description="Dropout rate")
    # drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    name: str = Field(default="MobileOne", description="Model name")


def mobileone_block(
    output_filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    groups: int = 1,
    dilation: int = 1,
    inference_mode: bool = False,
    se_ratio: int = 0,
    num_conv_branches: int = 1,
    activation: str = "relu6",
    name: str | None = None,
) -> keras.Layer:
    """MBConv block w/ expansion and SE

    Args:
        output_filters (int): Number of output filter channels
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 8.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        keras.Layer: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1]
        stride_len = strides if isinstance(strides, int) else sum(strides) / len(strides)
        kernel_len = kernel_size if isinstance(kernel_size, int) else sum(kernel_size) / len(kernel_size)
        is_downsample = stride_len > 1
        is_depthwise = groups > 1 and groups == input_filters
        has_skip_branch = output_filters == input_filters and stride_len == 1

        if inference_mode:
            y = keras.layers.ZeroPadding2D(padding=padding)(x)
            y = conv2d(
                output_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                dilation=dilation,
                groups=groups,
                use_bias=True,
                name=name,
            )(y)
            if se_ratio > 0:
                name_se = f"{name}_se" if name else None
                y = se_layer(ratio=se_ratio, name=name_se)(y)
            # END IF
            y = keras.layers.Activation(activation, name=f"{name}_act" if name else None)(y)
            return y
        # END IF

        branches = []

        # Skip branch
        if has_skip_branch:
            name_skip = f"{name}_skip" if name else None
            y_skip = batch_normalization(name=name_skip)(x)
            branches.append(y_skip)
        # END IF

        # Either groups is input_filters or is 1

        # Scale branch
        if kernel_len > 1:
            name_scale = f"{name}_scale" if name else None
            if is_depthwise:
                y_scale = keras.layers.DepthwiseConv2D(
                    kernel_size=(1, 1),
                    strides=(1, 1),  # strides,
                    padding="valid",
                    use_bias=False,
                    depthwise_initializer="he_normal",
                    name=f"{name_scale}_conv" if name_scale else None,
                )(x)
                y_scale = batch_normalization(name=name_scale)(y_scale)
                if is_downsample:
                    y_scale = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y_scale)
            else:
                y_scale = keras.layers.Conv2D(
                    output_filters,
                    kernel_size=(1, 1),
                    strides=strides,
                    padding="valid",
                    groups=groups,
                    use_bias=False,
                    kernel_initializer="he_normal",
                    name=f"{name_scale}_conv" if name_scale else None,
                )(x)
                y_scale = batch_normalization(name=name_scale)(y_scale)
            branches.append(y_scale)
        # END IF

        # Other branches
        yp = keras.layers.ZeroPadding2D(padding=padding)(x)
        for b in range(num_conv_branches):
            name_branch = f"{name}_branch{b + 1}" if name else None
            if is_depthwise:
                y_branch = keras.layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    padding="valid",
                    use_bias=False,
                    depthwise_initializer="he_normal",
                    name=f"{name_branch}_conv" if name else None,
                )(yp)
                y_branch = batch_normalization(name=name_branch)(y_branch)
                if is_downsample:
                    y_branch = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y_branch)
            else:
                y_branch = keras.layers.Conv2D(
                    output_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    groups=groups,
                    padding="valid",
                    use_bias=False,
                    kernel_initializer="he_normal",
                    name=f"{name_branch}_conv" if name else None,
                )(yp)
                y_branch = batch_normalization(name=name_branch)(y_branch)
            branches.append(y_branch)
        # END FOR

        # Merge branches
        y = keras.layers.Add(name=f"{name}_add" if name else None)(branches)

        # Squeeze-Excite block
        if se_ratio > 0:
            name_se = f"{name}_se" if name else None
            y = se_layer(ratio=se_ratio, name=name_se)(y)
        # END IF
        y = keras.layers.Activation(activation, name=f"{name}_act" if name else None)(y)
        return y

    # END DEF
    return layer


def mobileone_layer(
    x: keras.KerasTensor,
    params: MobileOneParams,
    num_classes: int | None = None,
    inference_mode: bool = False,
) -> keras.KerasTensor:
    """Create MobileOne TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (MobileOneParams): Model parameters.
        num_classes (int, optional): Number of classes.

    Returns:
        keras.KerasTensor: Output tensor
    """

    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x
    # END IF

    y = mobileone_block(
        output_filters=params.input_filters,
        kernel_size=params.input_kernel_size,
        strides=params.input_strides,
        padding=params.input_padding,
        groups=1,
        inference_mode=inference_mode,
        name=f"M0_B{0}_D{0}_DW",
    )(y)

    for b, block in enumerate(params.blocks):
        for d in range(block.depth):
            se_ratio = block.se_ratio if d >= block.depth - block.se_depth else 0
            # Depthwise block
            y = mobileone_block(
                output_filters=y.shape[-1],
                kernel_size=block.kernel_size,
                strides=block.strides if d == 0 else (1, 1),
                padding=block.padding,
                groups=y.shape[-1],
                inference_mode=inference_mode,
                se_ratio=se_ratio,
                num_conv_branches=block.num_conv_branches,
                name=f"M1_B{b + 1}_D{d + 1}_DW",
            )(y)

            # Pointwise block
            y = mobileone_block(
                output_filters=block.filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=(0, 0),
                groups=1,
                inference_mode=inference_mode,
                se_ratio=se_ratio,
                num_conv_branches=block.num_conv_branches,
                name=f"M1_B{b + 1}_D{d + 1}_PW",
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

    if requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


class MobileOneModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: MobileOneParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = MobileOneParams(**params)
        return mobileone_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: MobileOneParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = MobileOneModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
