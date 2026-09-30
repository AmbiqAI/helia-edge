# Copyright 2015 The TensorFlow Authors.
# Copyright (c) 2022 STMicroelectronics.
# Licensed under Apache-2.0; see licenses/miniresnet-apache-2.0.txt.
"""MiniResNet-v1 for channels-last spectrogram patches.

Adapted from ST's MiniResNet-v1 (services revision
0f6210ed5156126b782e1c43249063a477484b20). The default architecture matches
its one-stack ESC-10 checkpoint with input (64, 50, 1) and ten classes.
Constructors initialize weights; callers explicitly load trained weights using
Keras, and own audio preprocessing, seeds, class labels and export policy.
"""

import keras

from .miniresnet_params import MiniResNetV1Params

keras.saving.register_keras_serializable(package="helia_edge")(MiniResNetV1Params)


def _block(inputs: keras.KerasTensor, filters: int, *, projection: bool, name: str) -> keras.KerasTensor:
    stride = 2 if projection else 1
    shortcut = inputs
    if projection:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride, data_format="channels_last", name=f"{name}_0_conv")(
            shortcut
        )
        shortcut = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f"{name}_0_bn")(shortcut)
    x = keras.layers.Conv2D(filters, 1, strides=stride, data_format="channels_last", name=f"{name}_1_conv")(inputs)
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f"{name}_1_bn")(x)
    x = keras.layers.Activation("relu", name=f"{name}_1_relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same", data_format="channels_last", name=f"{name}_2_conv")(x)
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=f"{name}_2_bn")(x)
    x = keras.layers.Activation("relu", name=f"{name}_2_relu")(x)
    x = keras.layers.Add(name=f"{name}_add")([shortcut, x])
    return keras.layers.Activation("relu", name=f"{name}_out")(x)


class MiniResNetV1Model:
    """Build a standard Keras Functional model from typed architecture config."""

    @staticmethod
    def model_from_params(inputs: keras.KerasTensor, params: MiniResNetV1Params, num_classes: int) -> keras.Model:
        """Construct an untrained classifier for NHWC spectrogram patches.

        Hydration is explicit: ``model.load_weights(checkpoint_path)``. Only
        matching architecture, input dimensions and class count can reuse a
        checkpoint. This method does not read files or alter global settings.
        """
        if not isinstance(params, MiniResNetV1Params):
            raise TypeError("params must be MiniResNetV1Params; use from_config for mappings")
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if not keras.backend.is_keras_tensor(inputs) or len(inputs.shape) != 4:
            raise ValueError("inputs must be a rank-4 NHWC Keras tensor")
        if inputs.shape[-1] is None or inputs.shape[-1] < 1:
            raise ValueError("inputs must have a known positive channel count")
        if any(dim is not None and dim < 1 for dim in inputs.shape[1:3]):
            raise ValueError("spatial dimensions must be positive")
        if params.pooling == "flatten" and any(dim is None for dim in inputs.shape[1:3]):
            raise ValueError("flatten pooling requires fixed spatial dimensions")

        x = keras.layers.ZeroPadding2D(3, data_format="channels_last", name="conv1_pad")(inputs)
        x = keras.layers.Conv2D(params.base_filters, 7, strides=2, data_format="channels_last", name="conv1_conv")(x)
        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name="conv1_bn")(x)
        x = keras.layers.Activation("relu", name="conv1_relu")(x)
        x = keras.layers.ZeroPadding2D(1, data_format="channels_last", name="pool1_pad")(x)
        x = keras.layers.MaxPooling2D(3, strides=2, data_format="channels_last", name="pool1_pool")(x)
        for index in range(params.stacks):
            filters = params.base_filters * 2**index
            x = _block(x, filters, projection=True, name=f"conv{index + 2}_block1")
            x = _block(x, filters, projection=False, name=f"conv{index + 2}_block2")
        if params.pooling == "flatten":
            x = keras.layers.Flatten(data_format="channels_last", name="flatten")(x)
        elif params.pooling == "avg":
            x = keras.layers.GlobalAveragePooling2D(data_format="channels_last", name="avg_pool")(x)
        else:
            x = keras.layers.GlobalMaxPooling2D(data_format="channels_last", name="max_pool")(x)
        if params.dropout:
            x = keras.layers.Dropout(params.dropout, name="head_dropout")(x)
        outputs = keras.layers.Dense(num_classes, activation=params.output_activation, name="new_head")(x)
        return keras.Model(inputs=inputs, outputs=outputs, name=params.name)
