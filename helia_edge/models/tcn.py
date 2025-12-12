"""
# Temporal Convolutional Network (TCN)

## Overview

Temporal convolutional network (TCN) is a type of convolutional neural network (CNN) that is commonly used for sequence modeling tasks such as speech recognition, text generation, and video classification. TCN is a fully convolutional network that consists of a series of dilated causal convolutional layers. The dilated convolutional layers allow TCN to have a large receptive field while maintaining a small number of parameters. TCN is also fully parallelizable, which allows for faster training and inference times.

For more info, refer to the original paper [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://doi.org/10.48550/arXiv.1608.08242).

Classes:
    TcnParams: TCN parameters
    TcnBlockParams: TCN block parameters
    TcnModel: Helper class to generate model from parameters

Functions:
    normalization: Normalization layer
    tcn_block_lg: TCN large block
    tcn_block_mb: TCN mbconv block
    tcn_block_sm: TCN small block
    tcn_core: TCN core
    tcn_layer: TCN functional layer

## Additions

The TCN architecture has been modified to allow the following:

* Convolutional pairs can be factorized into depthwise separable convolutions.
* Squeeze and excitation (SE) blocks can be added between convolutional pairs.
* Normalization can be set between batch normalization and layer normalization.
* ReLU is replaced with the approximated ReLU6.

## Usage

The following example demonstrates how to create a TCN model using the `Tcn` class. The model is defined using a set of parameters defined in the `TcnParams` and `TcnBlockParams` classes.

```python
import keras
import helia_edge as helia

inputs = keras.Input(shape=(800, 1), name="inputs")
num_classes = 5

params = helia.models.TcnParams(
    input_kernel=(1, 3),
    input_norm="batch",
    blocks=[
        helia.models.TcnBlockParams(filters=8, kernel=(1, 3), dilation=(1, 1), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
        helia.models.TcnBlockParams(filters=16, kernel=(1, 3), dilation=(1, 2), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
        helia.models.TcnBlockParams(filters=24, kernel=(1, 3), dilation=(1, 4), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
        helia.models.TcnBlockParams(filters=32, kernel=(1, 3), dilation=(1, 8), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
    ],
    output_kernel=(1, 3),
    include_top=True,
    use_logits=True,
    model_name="tcn",
)

model = helia.models.TcnModel.model_from_params(
    inputs=inputs,
    params=params,
    num_classes=num_classes,
)
```

"""

from typing import Literal

import keras
from pydantic import BaseModel, Field

from ..layers.squeeze_excite import se_layer


class TcnBlockParams(BaseModel):
    """TCN block parameters

    Attributes:
        depth (int): Layer depth
        branch (int): Number of branches
        filters (int): Number of filters
        kernel (int | tuple[int, int]): Kernel size
        dilation (int | tuple[int, int]): Dilation rate
        ex_ratio (float): Expansion ratio
        se_ratio (float): Squeeze and excite ratio
        dropout (float | None): Dropout rate
        norm (Literal["batch", "layer"] | None): Normalization type
        activation (str): Activation function
    """

    depth: int = Field(default=1, description="Layer depth")
    branch: int = Field(default=1, description="Number of branches")
    filters: int = Field(..., description="# filters")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    dilation: int | tuple[int, int] = Field(default=1, description="Dilation rate")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    se_ratio: float = Field(default=0, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    activation: str = Field(default="relu6", description="Activation function")


class TcnParams(BaseModel):
    """TCN parameters

    Attributes:
        input_kernel (int | tuple[int, int] | None): Input kernel size
        input_norm (Literal["batch", "layer"] | None): Input normalization type
        block_type (Literal["lg", "mb", "sm"]): Block type
        blocks (list[TcnBlockParams]): TCN blocks
        output_kernel (int | tuple[int, int]): Output kernel size
        include_top (bool): Include top
        use_logits (bool): Use logits
        output_activation (str | None): Output activation
        name (str): Model name
    """

    input_kernel: int | tuple[int, int] | None = Field(default=None, description="Input kernel size")
    input_norm: Literal["batch", "layer"] | None = Field(default="layer", description="Input normalization type")
    block_type: Literal["lg", "mb", "sm"] = Field(default="mb", description="Block type")
    blocks: list[TcnBlockParams] = Field(default_factory=list, description="TCN blocks")
    output_kernel: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="TCN", description="Model name")


def normalization(norm: str, name: str) -> keras.Layer:
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
            return keras.layers.LayerNormalization(axis=(1, 2), name=f"{name}_LN")(x)
        return x

    return layer


def tcn_block_lg(params: TcnBlockParams, name: str) -> keras.Layer:
    """TCN large block

    Args:
        params (TcnBlockParams): Parameters
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """TCN block layer"""
        y = x

        for d in range(params.depth):
            lcl_name = f"{name}_D{d + 1}"
            y_skip = y

            y = keras.layers.Conv2D(
                filters=params.filters,
                kernel_size=params.kernel,
                strides=(1, 1),
                padding="same",
                use_bias=False,
                dilation_rate=params.dilation,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                name=f"{lcl_name}_CN1",
            )(y)
            y = normalization(params.norm, f"{lcl_name}_CN1")(y)

            y = keras.layers.Conv2D(
                filters=params.filters,
                kernel_size=params.kernel,
                strides=(1, 1),
                padding="same",
                use_bias=params.norm is None,
                dilation_rate=params.dilation,
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                name=f"{lcl_name}_CN2",
            )(y)
            y = normalization(params.norm, f"{lcl_name}_CN2")(y)

            if y_skip.shape[-1] == y.shape[-1]:
                y = keras.layers.Add(name=f"{lcl_name}_ADD")([y, y_skip])

            y = keras.layers.Activation(params.activation, name=f"{lcl_name}_ACT")(y)

            # Squeeze and excite
            if params.se_ratio > 0:
                y = se_layer(ratio=params.se_ratio, name=f"{lcl_name}_SE")(y)
            # END IF

            if params.dropout and params.dropout > 0:
                y = keras.layers.SpatialDropout2D(rate=params.dropout, name=f"{lcl_name}_DROP")(y)
            # END IF

        # END FOR
        return y

    return layer


def tcn_block_mb(params: TcnBlockParams, name: str) -> keras.Layer:
    """TCN mbconv block

    Args:
        params (TcnBlockParams): Parameters
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """TCN block layer"""
        y = x
        y_skip = y
        for d in range(params.depth):
            lcl_name = f"{name}_D{d + 1}"

            if params.ex_ratio != 1:
                y = keras.layers.Conv2D(
                    filters=int(params.filters * params.ex_ratio),
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}_EX_CN",
                )(y)
                y = normalization(params.norm, f"{lcl_name}_EX")(y)
                y = keras.layers.Activation(params.activation, name=f"{lcl_name}_EX_ACT")(y)
            # END IF

            branches = []
            for b in range(params.branch):
                yb = y
                yb = keras.layers.DepthwiseConv2D(
                    kernel_size=params.kernel,
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    dilation_rate=params.dilation,
                    depthwise_initializer="he_normal",
                    depthwise_regularizer=keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}_DW_B{b + 1}_CN",
                )(yb)
                yb = normalization(params.norm, f"{lcl_name}_DW_B{b + 1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = keras.layers.Add(name=f"{lcl_name}_DW_ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = keras.layers.Activation(params.activation, name=f"{lcl_name}_DW_ACT")(y)

            # Squeeze and excite
            if params.se_ratio and y.shape[-1] // params.se_ratio > 0:
                y = se_layer(ratio=params.se_ratio, name=f"{lcl_name}_SE")(y)
            # END IF

            branches = []
            for b in range(params.branch):
                yb = y
                yb = keras.layers.Conv2D(
                    filters=params.filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}_PW_B{b + 1}_CN",
                )(yb)
                yb = normalization(params.norm, f"{lcl_name}_PW_B{b + 1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = keras.layers.Add(name=f"{lcl_name}_PW_ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = keras.layers.Activation(params.activation, name=f"{lcl_name}_PW_ACT")(y)
        # END FOR

        # Skip connection
        if y_skip.shape[-1] == y.shape[-1]:
            y = keras.layers.Add(name=f"{name}_ADD")([y, y_skip])

        if params.dropout and params.dropout > 0:
            y = keras.layers.SpatialDropout2D(rate=params.dropout, name=f"{name}_DROP")(y)
        # END IF
        return y

    return layer


def tcn_block_sm(params: TcnBlockParams, name: str) -> keras.Layer:
    """TCN small block

    Args:
        params (TcnBlockParams): Parameters
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """TCN block layer"""
        y = x
        y_skip = y
        for d in range(params.depth):
            lcl_name = f"{name}_D{d + 1}"
            branches = []
            for b in range(params.branch):
                yb = y
                yb = keras.layers.DepthwiseConv2D(
                    kernel_size=params.kernel,
                    strides=(1, 1),
                    padding="same",
                    use_bias=params.norm is None,
                    dilation_rate=params.dilation,
                    depthwise_initializer="he_normal",
                    depthwise_regularizer=keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}_DW_B{b + 1}_CN",
                )(yb)
                yb = normalization(params.norm, f"{lcl_name}_DW_B{b + 1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = keras.layers.Add(name=f"{lcl_name}_DW_ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = keras.layers.Activation(params.activation, name=f"{lcl_name}_DW_ACT")(y)

            branches = []
            for b in range(params.branch):
                yb = y
                yb = keras.layers.Conv2D(
                    filters=params.filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    # groups=int(params.se_ratio) if params.se_ratio > 0 else 1,
                    use_bias=params.norm is None,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    name=f"{lcl_name}_PW_B{b + 1}_CN",
                )(yb)
                yb = normalization(params.norm, f"{lcl_name}_PW_B{b + 1}")(yb)
                branches.append(yb)
            # END FOR

            if params.branch > 1:
                y = keras.layers.Add(name=f"{lcl_name}_PW_ADD")(branches)
            else:
                y = branches[0]
            # END IF

            y = keras.layers.Activation(params.activation, name=f"{lcl_name}_PW_ACT")(y)
        # END FOR

        # Squeeze and excite
        if y.shape[-1] // params.se_ratio > 1:
            y = se_layer(ratio=params.se_ratio, name=f"{name}_SE")(y)
        # END IF

        # Skip connection
        if y_skip.shape[-1] == y.shape[-1]:
            y = keras.layers.Add(name=f"{name}_ADD")([y, y_skip])

        if params.dropout and params.dropout > 0:
            y = keras.layers.SpatialDropout2D(rate=params.dropout, name=f"{name}_DROP")(y)
        # END IF
        return y

    return layer


def tcn_core(params: TcnParams) -> keras.Layer:
    """TCN core

    Args:
        params (TcnParams): Parameters

    Returns:
        keras.Layer: Layer
    """
    if params.block_type == "lg":
        tcn_block = tcn_block_lg
    elif params.block_type == "mb":
        tcn_block = tcn_block_mb
    elif params.block_type == "sm":
        tcn_block = tcn_block_sm
    else:
        raise ValueError(f"Invalid block type: {params.block_type}")

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        for i, block in enumerate(params.blocks):
            name = f"B{i + 1}"
            y = tcn_block(params=block, name=name)(y)
        # END IF
        return y

    return layer


def tcn_layer(
    x: keras.KerasTensor,
    params: TcnParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """TCN functional layer

    Args:
        x (keras.KerasTensor): Input tensor
        params (TcnParams): Parameters
        num_classes (int): Number of classes

    Returns:
        keras.KerasTensor: Output tensor
    """
    requires_reshape = len(x.shape) == 3
    if requires_reshape:
        y = keras.layers.Reshape((1,) + x.shape[1:])(x)
    else:
        y = x

    ### Input layer
    # Encode each channel separately
    if params.input_kernel:
        y = keras.layers.DepthwiseConv2D(
            kernel_size=params.input_kernel,
            use_bias=params.input_norm is None,
            name="ENC_CN",
            padding="same",
        )(y)
        y = normalization(params.input_norm, "ENC")(y)
    # END IF

    ### Core layers
    y = tcn_core(params)(y)

    ### Output layer
    if params.include_top:
        # Add a per-point classification layer
        y = keras.layers.Conv2D(
            num_classes,
            kernel_size=params.output_kernel,
            padding="same",
            name="NECK_conv",
            use_bias=True,
        )(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
        elif not params.use_logits:
            y = keras.layers.Softmax()(y)
        # END IF
    # END IF

    # Always reshape back to original shape
    if requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)

    return y


class TcnModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: TcnParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = TcnParams(**params)
        return tcn_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: TcnParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = TcnModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
