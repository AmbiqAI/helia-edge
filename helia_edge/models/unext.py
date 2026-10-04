"""
# U-NeXt

## Overview

U-NeXt is a modification of U-Net that utilizes techniques from ResNeXt and EfficientNetV2. During the encoding phase, mbconv blocks are used to efficiently process the input.

Parameters are in ``helia_edge.models.unext_params``.

Functions:
    build: U-NeXt model from ``UNextParams``
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

from ..layers.normalization import LayerNormalization
from .unext_params import UNextParams


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
            return LayerNormalization(axis=ln_axis, name=f"{name}_LN")(x)
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
            y = LayerNormalization(
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
            y = LayerNormalization(
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
                y = LayerNormalization(
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


def unext_layer(inputs: keras.KerasTensor, params: UNextParams) -> keras.KerasTensor:
    """Create UNext TF functional model

    Args:
        inputs (keras.KerasTensor): Input tensor
        params (UNextParams): Model parameters.

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
        if params.num_classes is None:
            raise ValueError("UNext needs num_classes with include_top")
        # Add a per-point classification layer
        y = keras.layers.Conv2D(
            params.num_classes,
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


def build(
    params: UNextParams, input_shape: tuple[int | None, ...], *, batch_size: int | None = None, name: str | None = None
) -> keras.Model:
    """Build a UNext model.

    Args:
        params (UNextParams): Model parameters.
        input_shape (tuple[int | None, ...]): Input shape without the batch axis; None for a variable axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``unext`` unless ``name`` is given.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=unext_layer(inputs, params), name=name or params.family)
