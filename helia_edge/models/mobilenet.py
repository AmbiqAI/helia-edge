"""
# MobileNet Models API

This module provides utility functions to generate MobileNet models.

Classes:
    MobileNetV1Params: MobileNetV1 parameters
    MobileNetV1Model: Helper class to generate model from parameters

Functions:
    mobilenetv1_layer: Modified MobileNetV1 layer

"""

from typing import cast

import keras
from pydantic import BaseModel, Field


class MobileNetV1Params(BaseModel):
    """MobileNetV1 parameters

    Attributes:
        input_filters (int): Input filters
        input_strides (int | tuple[int, int]): Input stride
        include_top (bool): Include top
        output_activation (str | None): Output activation
        name (str): Model name
    """

    input_filters: int = Field(default=8, description="Input filters")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="RegNet", description="Model name")


def mobilenetv1_layer(
    x: keras.KerasTensor,
    num_classes: int,
    params: MobileNetV1Params,
) -> keras.KerasTensor:
    """Modified MobileNetV1
    MLPerf Tiny model for VWW:
    https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py

    Args:
        x (keras.KerasTensor): Model input
        num_classes (int): Number of classes.
        params (MobileNetV1Params): Model parameters

    Returns:
        keras.KerasTensor: Model output
    """
    num_filters = params.input_filters

    # 1st layer, pure conv
    # Keras 2.2 model has padding='valid' and disables bias
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=params.input_strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)  # Keras uses ReLU6 instead of pure ReLU

    # 2nd layer, depthwise separable conv
    # Filter size is always doubled before the pointwise conv
    # Keras uses ZeroPadding2D() and padding='valid'
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 3rd layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 4th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 5th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 6th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 7th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 8th-12th layers, identical depthwise separable convs
    for _ in range(8, 13):
        y = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

    # 13th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 14th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = cast(keras.KerasTensor, keras.layers.Activation("relu")(y))

    # Average pooling, max polling may be used also
    # Keras employs GlobalAveragePooling2D
    y = keras.layers.AveragePooling2D(pool_size=y.shape[1:3])(y)
    # x = MaxPooling2D(pool_size=x.shape[1:3])(x)

    # Keras inserts Dropout() and a pointwise Conv2D() here
    # We are staying with the paper base structure

    # Flatten, FC layer and classify
    if params.include_top:
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(num_classes)(y)

    return y


class MobileNetV1Model:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: MobileNetV1Params | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = MobileNetV1Params(**params)
        return mobilenetv1_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: MobileNetV1Params | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = MobileNetV1Model.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
