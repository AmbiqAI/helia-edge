"""
# MobileOne

## Overview

MobileOne is a fully convolutional neural network designed to have minimal latency when running in mobile/edge devices. The architecture consists of a series of depthwise separable convolutions and squeeze and excitation (SE) blocks. The network also uses standard batch normalization and ReLU activations that can be easily fused into the convolutional layers. Lastly, the network uses over-parameterized branches to improve training, yet can be merged into a single branch during inference.

For more info, refer to the original paper [MobileOne: An Improved One millisecond Mobile Backbone](https://doi.org/10.48550/arXiv.2206.04040).

Parameters are in ``helia_edge.models.mobileone_params``.

Functions:
    build: MobileOne model from ``MobileOneParams``
    mobileone_block: MobileOne block
    mobileone_layer: MobileOne layer

---

## Additions

The MobileOne architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Enable dilated convolutions.

## Usage

```python
from helia_edge.models import MobileOneBlockParams, MobileOneParams, ModelSpec, build

params = MobileOneParams(
    input_filters=24,
    input_kernel_size=(1, 7),
    input_strides=(1, 2),
    input_padding=(0, 3),
    blocks=[
        MobileOneBlockParams(
            filters=32, depth=2, kernel_size=(1, 7), strides=(1, 2), padding=(0, 3), se_ratio=2, num_conv_branches=2
        )
    ],
    include_top=True,
    num_classes=5,
)
model = build(ModelSpec(params=params, input_shape=(1, 800, 1)))
model.summary()
```

"""

import keras

from ..layers import se_layer
from ..layers.convolutional import conv2d
from ..layers.normalization import batch_normalization
from .mobileone_params import MobileOneParams


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


def mobileone_layer(x: keras.KerasTensor, params: MobileOneParams, inference_mode: bool = False) -> keras.KerasTensor:
    """Create MobileOne TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (MobileOneParams): Model parameters.

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
        if params.num_classes is not None:
            y = keras.layers.Dense(params.num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    if requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


def build(params: MobileOneParams, input_shape: tuple[int, ...], *, batch_size: int | None = None) -> keras.Model:
    """Build a MobileOne model.

    Args:
        params (MobileOneParams): Model parameters.
        input_shape (tuple[int, ...]): Input shape without the batch axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.

    Returns:
        keras.Model: The model, named ``mobileone``.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=mobileone_layer(inputs, params), name=params.family)
