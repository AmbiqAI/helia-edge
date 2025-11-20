"""
# EfficientNetV2

## Overview

EfficientNetV2 is an improvement to EfficientNet that incorporates additional optimizations to reduce both computation and memory.
In particular, the architecture leverages both fused and non-fused MBConv blocks, non-uniform layer scaling, and training-aware NAS.

For more info, refer to the original paper [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

Classes:
    EfficientNetV2Params: EfficientNetV2 parameters
    EfficientNetV2Model: Helper class to generate model from parameters

Functions:
    efficientnetv2_layer: EfficientNetV2 layer


## Additions

The EfficientNetV2 architecture has been modified to allow the following:

* Enable 1D and 2D variants.

## Usage

```python
import keras
from helia_edge.models import EfficientNetV2, EfficientNetV2Params, MBConvParams

inputs = keras.Input(shape=(800, 1))
num_classes = 5

model = EfficientNetV2(
    x=inputs,
    params=EfficientNetV2Params(
        input_filters=24,
        input_kernel_size=(1, 7),
        input_strides=(1, 2),
        blocks=[
            MBConvParams(filters=32, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
            MBConvParams(filters=48, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
            MBConvParams(filters=64, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
            MBConvParams(filters=72, depth=1, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2)
        ],
        output_filters=0,
        include_top=True,
        dropout=0.2,
        drop_connect_rate=0.2,
        model_name="efficientnetv2"
    ),
    num_classes=num_classes,
)


```

"""

from typing import Literal

import keras
from pydantic import BaseModel, Field

from ..layers.convolutional import conv2d
from ..layers.normalization import batch_normalization
from ..layers.mbconv import MBConvParams, mbconv_block
from .utils import make_divisible


class EfficientNetParams(BaseModel):
    """EfficientNet parameters

    Attributes:
        blocks (list[MBConvParams]): EfficientNet blocks
        input_filters (int): Input filters
        input_kernel_size (int | tuple[int, int]): Input kernel size
        input_strides (int | tuple[int, int]): Input stride
        input_activation (str): Input activation
        output_filters (int): Output filters
        output_activation (str | None): Output activation
        include_top (bool): Include top
        dropout (float): Dropout rate
        drop_connect_rate (float): Drop connect rate
        use_logits (bool): Use logits
        activation (str): Activation function
        norm (Literal["batch", "layer"] | None): Normalization type
        name (str): Model name
    """

    blocks: list[MBConvParams] = Field(default_factory=list, description="EfficientNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_activation: str = Field(default="relu6", description="Input activation")
    output_filters: int = Field(default=0, description="Output filters")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    use_logits: bool = Field(default=True, description="Use logits")
    activation: str = Field(default="relu6", description="Activation function")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    name: str = Field(default="EfficientNetV2", description="Model name")


def efficientnet_core(blocks: list[MBConvParams], drop_connect_rate: float = 0) -> keras.Layer:
    """EfficientNet core

    Args:
        blocks (list[MBConvParam]): MBConv params
        drop_connect_rate (float, optional): Drop connect rate. Defaults to 0.

    Returns:
        keras.Layer: Core
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        global_block_id = 0
        total_blocks = sum((b.depth for b in blocks))
        for i, block in enumerate(blocks):
            filters = make_divisible(block.filters, 8)
            for d in range(block.depth):
                name = f"stage{i+1}_mbconv{d+1}"
                block_drop_rate = drop_connect_rate * global_block_id / total_blocks
                x = mbconv_block(
                    filters,
                    block.ex_ratio,
                    block.kernel_size,
                    block.strides if d == 0 else 1,
                    block.se_ratio,
                    droprate=block_drop_rate,
                    bn_momentum=block.bn_momentum,
                    activation=block.activation,
                    name=name,
                )(x)
                global_block_id += 1
            # END FOR
        # END FOR
        return x

    # END DEF
    return layer


def efficientnetv2_layer(
    x: keras.KerasTensor,
    params: EfficientNetParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """Create EfficientNet V2 TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (EfficientNetParams): Model parameters.
        num_classes (int, optional): Number of classes.

    Returns:
        keras.KerasTensor: Output tensor
    """

    # Force input to be 4D (add dummy dimension)
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
        y = conv2d(
            filters,
            kernel_size=params.input_kernel_size,
            strides=params.input_strides,
            name=name,
        )(y)
        y = batch_normalization(name=name)(y)
        y = keras.layers.Activation(params.input_activation, name=f"{name}_act")(y)
    # END IF

    y = efficientnet_core(blocks=params.blocks, drop_connect_rate=params.drop_connect_rate)(y)

    if params.output_filters:
        name = "neck"
        filters = make_divisible(params.output_filters, 8)
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), padding="same", name=name)(y)
        y = batch_normalization(name=name)(y)
        y = keras.layers.Activation(params.output_activation, name=f"{name}_act")(y)

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}_pool")(y)
        if 0 < params.dropout < 1:
            y = keras.layers.Dropout(params.dropout)(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)

        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
        elif not params.use_logits:
            y = keras.layers.Softmax()(y)
    # Only reshape if required
    elif requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


class EfficientNetV2Model:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: EfficientNetParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = EfficientNetParams(**params)
        return efficientnetv2_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: EfficientNetParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = EfficientNetV2Model.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
