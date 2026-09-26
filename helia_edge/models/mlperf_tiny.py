"""Faithful scale-1 MLPerf Tiny architectures with newly initialized weights.

Pinned topology and licensing: docs/mlperf-tiny.md and models/licenses/.
Adapted from MLCommons Tiny (Apache-2.0); anomaly detector originally
Copyright (c) 2020 Hitachi, Ltd. (MIT). Changes: Keras 3 API, fixed captured
shapes/configurations, explicit layer names, no training or weight downloads.
These constructors do not provide official trained weights or accuracy claims.
"""

import keras


def _bn_relu(x, name):
    x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, name=f"{name}_bn")(x)
    return keras.layers.Activation("relu", name=f"{name}_relu")(x)


def mlperf_tiny_kws(*, name: str = "mlperf_tiny_kws") -> keras.Model:
    """DS-CNN64: 49x10x1 features to 12 softmax probabilities.

    Matches the pinned saved model's 25x5 final pool (not the training script's
    floor-based 24x5 pool). No audio preprocessing or trained weights included.
    """
    inputs = keras.Input(shape=(49, 10, 1), name="features")
    x = keras.layers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=True,
                            kernel_initializer="glorot_uniform", kernel_regularizer=keras.regularizers.L2(1e-4),
                            name="stem")(inputs)
    x = _bn_relu(x, "stem")
    x = keras.layers.Dropout(0.2, name="stem_dropout")(x)
    for stage in range(1, 5):
        # Serialized reference depthwise kernels use GlorotUniform and no regularizer.
        x = keras.layers.DepthwiseConv2D((3, 3), padding="same", depth_multiplier=1, use_bias=True,
                                        depthwise_initializer="glorot_uniform", name=f"block{stage}_dw")(x)
        x = _bn_relu(x, f"block{stage}_dw")
        x = keras.layers.Conv2D(64, (1, 1), padding="same", use_bias=True,
                                kernel_initializer="glorot_uniform", kernel_regularizer=keras.regularizers.L2(1e-4),
                                name=f"block{stage}_pw")(x)
        x = _bn_relu(x, f"block{stage}_pw")
    x = keras.layers.Dropout(0.4, name="head_dropout")(x)
    x = keras.layers.AveragePooling2D((25, 5), name="pool")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(12, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=name)


def mlperf_tiny_vww(*, name: str = "mlperf_tiny_vww") -> keras.Model:
    """Reference MobileNetV1 alpha0.25: 96x96 RGB to 2 softmax probabilities.

    Uses reference SAME padding, biases and ordinary ReLU, not application
    MobileNet's zero-padding/ReLU6 variant. No image preprocessing included.
    """
    inputs = keras.Input(shape=(96, 96, 3), name="image")
    x = keras.layers.Conv2D(8, 3, strides=2, padding="same", use_bias=True,
                            kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.L2(1e-4),
                            name="stem")(inputs)
    x = _bn_relu(x, "stem")
    stages = ((16, 1), (32, 2), (32, 1), (64, 2), (64, 1), (128, 2),
              (128, 1), (128, 1), (128, 1), (128, 1), (128, 1), (256, 2), (256, 1))
    for stage, (filters, stride) in enumerate(stages, 1):
        x = keras.layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=True,
                                        depthwise_initializer="glorot_uniform", name=f"block{stage}_dw")(x)
        x = _bn_relu(x, f"block{stage}_dw")
        x = keras.layers.Conv2D(filters, 1, padding="same", use_bias=True,
                                kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.L2(1e-4),
                                name=f"block{stage}_pw")(x)
        x = _bn_relu(x, f"block{stage}_pw")
    x = keras.layers.AveragePooling2D((3, 3), name="pool")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=name)


def mlperf_tiny_resnet(*, name: str = "mlperf_tiny_resnet") -> keras.Model:
    """Captured reduced ResNet: 32x32 RGB, widths28/56/112, 10 probabilities.

    Three residual stacks, nine convolutions; projection shortcuts have no
    batch normalization. This is not a generic ResNet18 constructor.
    """
    def conv(x, filters, kernel, stride, layer_name):
        return keras.layers.Conv2D(filters, kernel, strides=stride, padding="same", use_bias=True,
                                   kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.L2(1e-4),
                                   name=layer_name)(x)

    inputs = keras.Input(shape=(32, 32, 3), name="image")
    x = _bn_relu(conv(inputs, 28, 3, 1, "stem"), "stem")
    for stage, (filters, stride) in enumerate(((28, 1), (56, 2), (112, 2)), 1):
        y = _bn_relu(conv(x, filters, 3, stride, f"stack{stage}_conv1"), f"stack{stage}_conv1")
        y = conv(y, filters, 3, 1, f"stack{stage}_conv2")
        y = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                                             name=f"stack{stage}_conv2_bn")(y)
        if stride == 2:
            x = conv(x, filters, 1, 2, f"stack{stage}_projection")
        x = keras.layers.Add(name=f"stack{stage}_add")([x, y])
        x = keras.layers.Activation("relu", name=f"stack{stage}_relu")(x)
    x = keras.layers.AveragePooling2D((8, 8), name="pool")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(10, activation="softmax", kernel_initializer="he_normal", name="predictions")(x)
    return keras.Model(inputs, outputs, name=name)


def mlperf_tiny_ad(*, name: str = "mlperf_tiny_ad") -> keras.Model:
    """Dense AD01 autoencoder: 640 features to 640 linear reconstructions.

    Hidden widths128x4,8,128x4; BN/ReLU follows each hidden Dense, including
    the bottleneck. No anomaly score or audio preprocessing is computed.
    """
    inputs = keras.Input(shape=(640,), name="features")
    x = inputs
    for stage, units in enumerate((128, 128, 128, 128, 8, 128, 128, 128, 128), 1):
        x = keras.layers.Dense(units, use_bias=True, kernel_initializer="glorot_uniform", name=f"dense{stage}")(x)
        x = _bn_relu(x, f"dense{stage}")
    outputs = keras.layers.Dense(640, activation="linear", name="reconstruction")(x)
    return keras.Model(inputs, outputs, name=name)
