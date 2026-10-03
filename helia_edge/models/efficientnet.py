"""
# EfficientNetV2

## Overview

EfficientNetV2 is an improvement to EfficientNet that incorporates additional optimizations to reduce both computation and memory.
In particular, the architecture leverages both fused and non-fused MBConv blocks, non-uniform layer scaling, and training-aware NAS.

For more info, refer to the original paper [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).

Parameters are in ``helia_edge.models.efficientnet_params``.

Functions:
    build: EfficientNetV2 model from ``EfficientNetParams``
    efficientnetv2_layer: EfficientNetV2 layer


## Additions

The EfficientNetV2 architecture has been modified to allow the following:

* Enable 1D and 2D variants.

## Usage

```python
from helia_edge.layers import MBConvParams
from helia_edge.models import EfficientNetParams, ModelSpec, build

params = EfficientNetParams(
    input_filters=24,
    input_kernel_size=(1, 7),
    input_strides=(1, 2),
    blocks=[
        MBConvParams(filters=32, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
        MBConvParams(filters=48, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
        MBConvParams(filters=64, depth=2, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
        MBConvParams(filters=72, depth=1, kernel_size=(1, 7), strides=(1, 2), ex_ratio=1, se_ratio=2),
    ],
    output_filters=0,
    include_top=True,
    num_classes=5,
    dropout=0.2,
    drop_connect_rate=0.2,
)
model = build(ModelSpec(params=params, input_shape=(1, 800, 1)))
```

"""

import keras

from ..layers.convolutional import conv2d
from ..layers.mbconv import mbconv_block
from ..layers.mbconv_params import MBConvParams
from ..layers.normalization import batch_normalization
from .efficientnet_params import EfficientNetParams
from .utils import make_divisible


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
                name = f"stage{i + 1}_mbconv{d + 1}"
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


def efficientnetv2_layer(x: keras.KerasTensor, params: EfficientNetParams) -> keras.KerasTensor:
    """Create EfficientNet V2 TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (EfficientNetParams): Model parameters.

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
        if params.num_classes is not None:
            y = keras.layers.Dense(params.num_classes, name=name)(y)

        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
        elif not params.use_logits:
            y = keras.layers.Softmax()(y)
    # Only reshape if required
    elif requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


def build(params: EfficientNetParams, input_shape: tuple[int, ...], *, batch_size: int | None = None) -> keras.Model:
    """Build a EfficientNetV2 model.

    Args:
        params (EfficientNetParams): Model parameters.
        input_shape (tuple[int, ...]): Input shape without the batch axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.

    Returns:
        keras.Model: The model, named ``efficientnet``.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=efficientnetv2_layer(inputs, params), name=params.family)
