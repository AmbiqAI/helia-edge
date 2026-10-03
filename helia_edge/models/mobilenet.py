"""
# MobileNet Models API

This module provides utility functions to generate MobileNet models.

Parameters are in ``helia_edge.models.mobilenet_params``.

Functions:
    build: MobileNetV1 model from ``MobileNetV1Params``
    mobilenetv1_layer: Modified MobileNetV1 layer

"""

from typing import cast

import keras

from .mobilenet_params import MobileNetV1Params


def mobilenetv1_layer(x: keras.KerasTensor, params: MobileNetV1Params) -> keras.KerasTensor:
    """Modified MobileNetV1
    MLPerf Tiny model for VWW:
    https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py

    Args:
        x (keras.KerasTensor): Model input
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
            depthwise_initializer="he_normal",
            depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        depthwise_initializer="he_normal",
        depthwise_regularizer=keras.regularizers.l2(1e-4),
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
        y = keras.layers.Dense(params.num_classes)(y)

    return y


def build(params: MobileNetV1Params, input_shape: tuple[int, ...], *, batch_size: int | None = None) -> keras.Model:
    """Build a MobileNetV1 model.

    Args:
        params (MobileNetV1Params): Model parameters.
        input_shape (tuple[int, ...]): Input shape without the batch axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.

    Returns:
        keras.Model: The model, named ``mobilenet``.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=mobilenetv1_layer(inputs, params), name=params.family)
