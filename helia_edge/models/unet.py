"""
# U-Net

## Overview

U-Net is a type of convolutional neural network (CNN) that is commonly used for segmentation tasks. U-Net is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow U-Net to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to the original paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28).

Classes:
    UNetParams: U-Net parameters
    UNetModel: Helper class to generate model from parameters

Functions:
    unet_layer: Generate functional U-Net model


## Additions

The U-Net architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Convolutional pairs can factorized into depthwise separable convolutions.
* Specifiy the number of convolutional layers per block both downstream and upstream.
* Normalization can be set between batch normalization and layer normalization.
* ReLU is replaced with the approximated ReLU6.

## Usage

Instantiate from UNetParams:

```python
import keras
from helia_edge.models import UNet, UNetParams, UNetBlockParams

inputs = keras.Input(shape=(800, 1))
num_classes = 5

model = UNet(
    x=inputs,
    params=UNetParams(
        blocks=[
            UNetBlockParams(filters=12, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True),
            UNetBlockParams(filters=24, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True),
            UNetBlockParams(filters=32, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True),
            UNetBlockParams(filters=48, depth=2, ddepth=1, kernel=(1, 5), pool=(1, 3), strides=(1, 2), skip=True, seperable=True)
        ],
        output_kernel_size=(1, 5),
        include_top=True,
        use_logits=True,
        model_name="unet"
    ),
    num_classes=num_classes,
)
```

Instantiate from object:

```python


params = {
    "name": "unet",
    "params": {
        "blocks": [
            {"filters": 12, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true},
            {"filters": 24, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true},
            {"filters": 32, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true},
            {"filters": 48, "depth": 2, "ddepth": 1, "kernel": [1, 5], "pool": [1, 3], "strides": [1, 2], "skip": true, "seperable": true}
        ],
        "output_kernel_size": [1, 5],
        "include_top": true,
        "use_logits": true,
        "model_name": "efficientnetv2"
    }
}

model = unet_from_object(inputs, params, num_classes)
```


"""

from typing import Literal

import keras
from pydantic import BaseModel, Field

from ..layers.normalization import batch_normalization, layer_normalization


class UNetBlockParams(BaseModel):
    """UNet block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        ddepth (int | None): Decoder depth
        kernel (int | tuple[int, int]): Kernel size
        pool (int | tuple[int, int]): Pool size
        strides (int | tuple[int, int]): Stride size
        skip (bool): Add skip connection
        seperable (bool): Use seperable convs
        dropout (float | None): Dropout rate
        norm (Literal["batch", "layer"] | None): Normalization type
        activation (Literal["relu", "relu6", "leaky_relu", "elu", "selu"]): Activation
        dilation (int | tuple[int, int] | None): Dilation factor
    """

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Decoder depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=3, description="Pool size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    seperable: bool = Field(default=False, description="Use seperable convs")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="batch", description="Normalization type")
    activation: Literal["relu", "relu6", "leaky_relu", "elu", "selu"] = Field(default="relu6", description="Activation")
    dilation: int | tuple[int, int] | None = Field(default=None, description="Dilation factor")


class UNetParams(BaseModel):
    """UNet parameters

    Attributes:
        blocks (list[UNetBlockParams]): UNet blocks
        include_top (bool): Include top
        use_logits (bool): Use logits
        name (str): Model name
        output_kernel_size (int | tuple[int, int]): Output kernel size
        output_kernel_stride (int | tuple[int, int]): Output kernel stride
    """

    blocks: list[UNetBlockParams] = Field(default_factory=list, description="UNet blocks")
    include_top: bool = Field(default=True, description="Include top")
    use_logits: bool = Field(default=True, description="Use logits")
    name: str = Field(default="UNet", description="Model name")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    output_kernel_stride: int | tuple[int, int] = Field(default=1, description="Output kernel stride")


def unet_layer(
    x: keras.KerasTensor,
    params: UNetParams,
    num_classes: int,
) -> keras.KerasTensor:
    """Create UNet TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (ResNetParams): Model parameters.
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

    #### ENCODER ####
    skip_layers: list[keras.layers.Layer | None] = []
    for i, block in enumerate(params.blocks):
        name = f"ENC{i+1}"
        ym = y
        for d in range(block.depth):
            dname = f"{name}_D{d+1}"
            if block.dilation is None:
                dilation_rate = (1, 1)
            elif isinstance(block.dilation, int):
                dilation_rate = (block.dilation**d, block.dilation**d)
            else:
                dilation_rate = (block.dilation[0] ** d, block.dilation[1] ** d)
            if block.seperable:
                ym = keras.layers.SeparableConv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    depthwise_initializer="he_normal",
                    pointwise_initializer="he_normal",
                    depthwise_regularizer=keras.regularizers.L2(1e-3),
                    pointwise_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}_conv",
                )(ym)
            else:
                ym = keras.layers.Conv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}_conv",
                )(ym)
            if block.norm == "layer":
                ym = layer_normalization(name=dname, axis=[1, 2])(ym)
            elif block.norm == "batch":
                ym = batch_normalization(name=dname, momentum=0.99)(ym)
            ym = keras.layers.Activation(block.activation, name=f"{dname}_act")(ym)
        # END FOR

        # Project residual
        yr = keras.layers.Conv2D(
            block.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}_skip",
        )(y)

        if block.dropout is not None:
            ym = keras.layers.Dropout(block.dropout, noise_shape=ym.shape)(ym)
        y = keras.layers.add([ym, yr], name=f"{name}_add")

        skip_layers.append(y if block.skip else None)

        y = keras.layers.MaxPooling2D(block.pool, strides=block.strides, padding="same", name=f"{name}_pool")(y)
    # END FOR

    #### DECODER ####
    for i, block in enumerate(reversed(params.blocks)):
        name = f"DEC{i+1}"
        for d in range(block.ddepth or block.depth):
            dname = f"{name}_D{d+1}"
            if block.seperable:
                y = keras.layers.SeparableConv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    depthwise_initializer="he_normal",
                    pointwise_initializer="he_normal",
                    depthwise_regularizer=keras.regularizers.L2(1e-3),
                    pointwise_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}_conv",
                )(y)
            else:
                y = keras.layers.Conv2D(
                    block.filters,
                    kernel_size=block.kernel,
                    strides=(1, 1),
                    padding="same",
                    dilation_rate=dilation_rate,
                    kernel_initializer="he_normal",
                    kernel_regularizer=keras.regularizers.L2(1e-3),
                    use_bias=block.norm is None,
                    name=f"{dname}_conv",
                )(y)
            if block.norm == "layer":
                y = layer_normalization(name=dname, axis=[1, 2])(y)
            elif block.norm == "batch":
                y = batch_normalization(name=dname, momentum=0.99)(y)
            y = keras.layers.Activation(block.activation, name=f"{dname}_act")(y)
        # END FOR

        y = keras.layers.UpSampling2D(size=block.strides, name=f"{dname}_unpool")(y)

        # Add skip connection
        dname = f"{name}_D{block.depth+1}"
        skip_layer = skip_layers.pop()
        if skip_layer is not None:
            y = keras.layers.concatenate([y, skip_layer], name=f"{dname}_cat")  # Can add or concatenate
            # Use 1x1 conv to reduce filters
            y = keras.layers.Conv2D(
                block.filters,
                kernel_size=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{dname}_conv",
            )(y)
            if block.norm == "layer":
                y = layer_normalization(name=dname, axis=[1, 2])(y)
            elif block.norm == "batch":
                y = batch_normalization(name=dname, momentum=0.99)(y)
            y = keras.layers.Activation(block.activation, name=f"{dname}_act")(y)
        # END IF

        dname = f"{name}_D{block.depth+2}"
        if block.seperable:
            ym = keras.layers.SeparableConv2D(
                block.filters,
                kernel_size=block.kernel,
                strides=(1, 1),
                padding="same",
                depthwise_initializer="he_normal",
                pointwise_initializer="he_normal",
                depthwise_regularizer=keras.regularizers.L2(1e-3),
                pointwise_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{dname}_conv",
            )(y)
        else:
            ym = keras.layers.Conv2D(
                block.filters,
                kernel_size=block.kernel,
                strides=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.L2(1e-3),
                use_bias=block.norm is None,
                name=f"{dname}_conv",
            )(y)
        if block.norm == "layer":
            ym = layer_normalization(name=dname, axis=[1, 2])(ym)
        elif block.norm == "batch":
            ym = batch_normalization(name=dname, momentum=0.99)(ym)
        ym = keras.layers.Activation(block.activation, name=f"{dname}_act")(ym)

        # Project residual
        yr = keras.layers.Conv2D(
            block.filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.L2(1e-3),
            name=f"{name}_skip",
        )(y)
        y = keras.layers.add([ym, yr], name=f"{name}_add")  # Add back residual
    # END FOR

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

    if requires_reshape:
        y = keras.layers.Reshape(y.shape[2:])(y)
    # END IF

    return y


class UNetModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: UNetParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = UNetParams(**params)
        return unet_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: UNetParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = UNetModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
