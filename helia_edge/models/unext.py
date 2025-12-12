"""
# U-NeXt

## Overview

U-NeXt is a modification of U-Net that utilizes techniques from ResNeXt and EfficientNetV2. During the encoding phase, mbconv blocks are used to efficiently process the input.

Classes:
    UNextParams: U-NeXt parameters
    UNextModel: Helper class to generate

Functions:
    unext_block: Create U-NeXt block
    se_block: Squeeze and excite block
    norm_layer: Normalization layer
    unext_core: Create U-NeXt core
    unext_layer: Create U-NeXt layer

## Additions

The U-NeXt architecture has been modified to allow the following:

* MBConv blocks used in the encoding phase.
* Squeeze and excitation (SE) blocks added within blocks.

"""

from typing import Literal

import keras
from pydantic import BaseModel, Field


class UNextBlockParams(BaseModel):
    """UNext block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        ddepth (int | None): Layer decoder depth
        kernel (int | tuple[int, int]): Kernel size
        pool (int | tuple[int, int]): Pool size
        strides (int | tuple[int, int]): Stride size
        skip (bool): Add skip connection
        expand_ratio (float): Expansion ratio
        se_ratio (float): Squeeze and excite ratio
        dropout (float | None): Dropout rate
        norm (Literal["batch", "layer"] | None): Normalization type

    """

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Layer decoder depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=2, description="Pool size")
    strides: int | tuple[int, int] = Field(default=2, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    expand_ratio: float = Field(default=1, description="Expansion ratio")
    se_ratio: float = Field(default=0, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")


class UNextParams(BaseModel):
    """UNext parameters

    Attributes:
        blocks (list[UNextBlockParams]): UNext blocks
        include_top (bool): Include top
        use_logits (bool): Use logits
        output_kernel_size (int | tuple[int, int]): Output kernel size
        output_kernel_stride (int | tuple[int, int]): Output kernel stride
        name (str): Model name

    """

    blocks: list[UNextBlockParams] = Field(default_factory=list, description="UNext blocks")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    output_kernel_stride: int | tuple[int, int] = Field(default=1, description="Output kernel stride")
    name: str = Field(default="UNext", description="Model name")


def se_block(ratio: int = 8, name: str | None = None):
    """Squeeze and excite block"""

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        num_chan = x.shape[-1]
        # Squeeze
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}_pool" if name else None, keepdims=True)(x)

        y = keras.layers.Conv2D(
            num_chan // ratio,
            kernel_size=1,
            use_bias=True,
            name=f"{name}_sq" if name else None,
        )(y)

        y = keras.layers.Activation("relu6", name=f"{name}_relu" if name else None)(y)

        # Excite
        y = keras.layers.Conv2D(num_chan, kernel_size=1, use_bias=True, name=f"{name}_ex" if name else None)(y)
        y = keras.layers.Activation(keras.activations.hard_sigmoid, name=f"{name}_sigg" if name else None)(y)
        y = keras.layers.Multiply(name=f"{name}_mul" if name else None)([x, y])
        return y

    return layer


def norm_layer(norm: str, name: str) -> keras.Layer:
    """Normalization layer

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """Functional normalization layer

        Args:
            x (keras.KerasTensor): Input tensor

        Returns:
            keras.KerasTensor: Output tensor
        """
        if norm == "batch":
            return keras.layers.BatchNormalization(axis=-1, name=f"{name}_BN")(x)
        if norm == "layer":
            ln_axis = 2 if x.shape[1] == 1 else 1 if x.shape[2] == 1 else (1, 2)
            return keras.layers.LayerNormalization(axis=ln_axis, name=f"{name}_LN")(x)
        return x

    return layer


def unext_block(
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 4,
    dropout: float | None = 0,
    norm: Literal["batch", "layer"] | None = "batch",
    name: str | None = None,
) -> keras.Layer:
    """Create UNext block"""

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters: int = x.shape[-1]
        strides_len = strides if isinstance(strides, int) else sum(strides) // len(strides)
        add_residual = input_filters == output_filters and strides_len == 1
        ln_axis = 2 if x.shape[1] == 1 else 1 if x.shape[2] == 1 else (1, 2)

        # Depthwise conv
        y = keras.layers.Conv2D(
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            strides=1,
            padding="same",
            use_bias=norm is None,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}_dwconv" if name else None,
        )(x)
        if norm == "batch":
            y = keras.layers.BatchNormalization(
                name=f"{name}_norm",
            )(y)
        elif norm == "layer":
            y = keras.layers.LayerNormalization(
                axis=ln_axis,
                name=f"{name}_norm" if name else None,
            )(y)
        # END IF

        # Inverted expansion block
        if expand_ratio != 1:
            y = keras.layers.Conv2D(
                filters=int(expand_ratio * input_filters),
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=norm is None,
                groups=input_filters,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                name=f"{name}_expand" if name else None,
            )(y)

            y = keras.layers.Activation(
                "relu6",
                name=f"{name}_relu" if name else None,
            )(y)

        # Squeeze and excite
        if se_ratio > 1:
            name_se = f"{name}_se" if name else None
            y = se_block(ratio=se_ratio, name=name_se)(y)

        y = keras.layers.Conv2D(
            filters=output_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=norm is None,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}_project" if name else None,
        )(y)

        if add_residual:
            if dropout and dropout > 0:
                y = keras.layers.Dropout(
                    dropout,
                    noise_shape=(y.shape),
                    name=f"{name}_drop" if name else None,
                )(y)
            y = keras.layers.Add(name=f"{name}_res" if name else None)([x, y])
        return y

    # END DEF
    return layer


def unext_core(
    x: keras.KerasTensor,
    params: UNextParams,
) -> keras.KerasTensor:
    """Create UNext TF functional core

    Args:
        x (keras.KerasTensor): Input tensor
        params (UNextParams): Model parameters.

    Returns:
        keras.KerasTensor: Output tensor
    """

    y = x

    #### ENCODER ####
    skip_layers: list[keras.layers.Layer | None] = []
    for i, block in enumerate(params.blocks):
        name = f"ENC{i + 1}"
        for d in range(block.depth):
            y = unext_block(
                output_filters=block.filters,
                expand_ratio=block.expand_ratio,
                kernel_size=block.kernel,
                strides=1,
                se_ratio=block.se_ratio,
                dropout=block.dropout,
                norm=block.norm,
                name=f"{name}_D{d + 1}",
            )(y)
        # END FOR
        skip_layers.append(y if block.skip else None)

        # Downsample using strided conv
        y = keras.layers.Conv2D(
            filters=block.filters,
            kernel_size=block.pool,
            strides=block.strides,
            padding="same",
            use_bias=block.norm is None,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}_pool",
        )(y)
        if block.norm == "batch":
            y = keras.layers.BatchNormalization(
                name=f"{name}_norm",
            )(y)
        elif block.norm == "layer":
            ln_axis = 2 if y.shape[1] == 1 else 1 if y.shape[2] == 1 else (1, 2)
            y = keras.layers.LayerNormalization(
                axis=ln_axis,
                name=f"{name}_norm",
            )(y)
        # END IF
    # END FOR

    #### DECODER ####
    for i, block in enumerate(reversed(params.blocks)):
        name = f"DEC{i + 1}"
        for d in range(block.ddepth or block.depth):
            y = unext_block(
                output_filters=block.filters,
                expand_ratio=block.expand_ratio,
                kernel_size=block.kernel,
                strides=1,
                se_ratio=block.se_ratio,
                dropout=block.dropout,
                norm=block.norm,
                name=f"{name}_D{d + 1}",
            )(y)
        # END FOR

        # Upsample using transposed conv
        # y = keras.layers.Conv1DTranspose(
        #     filters=block.filters,
        #     kernel_size=block.pool,
        #     strides=block.strides,
        #     padding="same",
        #     kernel_initializer="he_normal",
        #     kernel_regularizer=keras.regularizers.L2(1e-3),
        #     name=f"{name}_unpool",
        # )(y)

        y = keras.layers.Conv2D(
            filters=block.filters,
            kernel_size=block.pool,
            strides=1,
            padding="same",
            use_bias=block.norm is None,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}_conv",
        )(y)
        y = keras.layers.UpSampling2D(size=block.strides, name=f"{name}_unpool")(y)

        # Skip connection
        skip_layer = skip_layers.pop()
        if skip_layer is not None:
            # y = keras.layers.Concatenate(name=f"{name}_S1_cat")([y, skip_layer])
            y = keras.layers.Add(name=f"{name}_S1_cat")([y, skip_layer])

            # Use conv to reduce filters
            y = keras.layers.Conv2D(
                block.filters,
                kernel_size=1,  # block.kernel,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{name}_S1_conv",
            )(y)

            if block.norm == "batch":
                y = keras.layers.BatchNormalization(
                    name=f"{name}_S1_norm",
                )(y)
            elif block.norm == "layer":
                ln_axis = 2 if y.shape[1] == 1 else 1 if y.shape[2] == 1 else (1, 2)
                y = keras.layers.LayerNormalization(
                    axis=ln_axis,
                    name=f"{name}_S1_norm",
                )(y)
            # END IF

            y = keras.layers.Activation(
                "relu6",
                name=f"{name}_S1_relu" if name else None,
            )(y)
        # END IF

        y = unext_block(
            output_filters=block.filters,
            expand_ratio=block.expand_ratio,
            kernel_size=block.kernel,
            strides=1,
            se_ratio=block.se_ratio,
            dropout=block.dropout,
            norm=block.norm,
            name=f"{name}_D{block.depth + 1}",
        )(y)

    # END FOR
    return y


def unext_layer(
    inputs: keras.KerasTensor,
    params: UNextParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """Create UNext TF functional model

    Args:
        inputs (keras.KerasTensor): Input tensor
        params (UNextParams): Model parameters.
        num_classes (int, optional): Number of classes.

    Returns:
        keras.KerasTensor: Output tensor
    """
    requires_reshape = len(inputs.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + inputs.shape[1:])(inputs)
    else:
        y = inputs

    y = unext_core(y, params)

    if params.include_top:
        # Add a per-point classification layer
        y = keras.layers.Conv2D(
            num_classes,
            kernel_size=params.output_kernel_size,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name="NECK_conv",
            use_bias=True,
        )(y)
        if not params.use_logits:
            y = keras.layers.Softmax()(y)
        # END IF
    # END IF

    # Always reshape back to original shape
    if requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


class UNextModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: UNextParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = UNextParams(**params)
        return unext_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: UNextParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = UNextModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
