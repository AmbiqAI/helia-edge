"""
# Normalization Layers API

This module provides classes to build normalization layers.

Functions:
    layer_normalization: Layer normalization
    batch_normalization: Batch normalization
    normalization: Normalization builder layer

Please check [Keras Normalization Layers](https://keras.io/api/layers/normalization_layers/) for additional layers.
"""

import keras


def layer_normalization(
    name: str | None = None, axis: int | tuple[int] | None = None, scale: bool = True
) -> keras.Layer:
    """Layer normalization

    If axis is None, this layer will infer based on the input tensor shape:

    * If rank is 4 (B, H, W, C), normalize over H, W
    * If rank is 3 (B, T, C), normalize over T
    * If rank is 2 (B, C), normalize over C

    Args:
        name (str | None, optional): Layer name. Defaults to None.
        axis (int | tuple[int] | None, optional): Axis. Defaults to None.
        scale (bool, optional): Scale. Defaults to True.

    Returns:
        keras.Layer: Layer
    """
    name = name + ".ln" if name else None

    if axis is None:

        def layer(x: keras.KerasTensor) -> keras.KerasTensor:
            """Functional layer normalization layer

            Args:
                x (tf.Tensor): Input tensor

            Returns:
                tf.Tensor: Output tensor
            """
            is_channels_first = keras.backend.image_data_format() == "channels_first"
            # If rank is 4 (B, H, W, C), normalize over H, W
            if x.shape.rank == 4:
                _axis = [-2, -3] if not is_channels_first else [-1, -2]
            # If rank is 3 (B, T, C), normalize over T
            elif x.shape.rank == 3:
                _axis = -2 if not is_channels_first else -1
            # If rank is 2 (B, C), normalize over C
            else:
                _axis = -1
            # END IF

            return keras.layers.LayerNormalization(axis=_axis, name=name, scale=scale)(x)

        # END DEF
    else:

        def layer(x: keras.KerasTensor) -> keras.KerasTensor:
            """Functional layer normalization layer

            Args:
                x (tf.Tensor): Input tensor

            Returns:
                tf.Tensor: Output tensor
            """
            return keras.layers.LayerNormalization(axis=axis, name=name, scale=scale)(x)

        # END DEF

    return layer


def batch_normalization(name: str | None = None, momentum=0.9, epsilon=1e-3, axis: int | None = -1) -> keras.Layer:
    """Batch normalization

    Args:
        name (str | None, optional): Layer name. Defaults to None.
        momentum (float, optional): Momentum. Defaults to 0.9.
        epsilon (float, optional): Epsilon. Defaults to 1e-3.
        axis (int | None, optional): Axis. Defaults to None.

    Returns:
        keras.Layer: Layer
    """
    name = name + "_bn" if name else None

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """Functional batch normalization layer

        Args:
            x (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """
        return keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, axis=axis, name=name)(x)

    # END DEF

    return layer


def normalization(norm: str, name: str, **kwargs) -> keras.Layer:
    """Creates normalization layer based on type

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        KerasLayer: Layer
    """
    if norm == "batch":
        return batch_normalization(name=name, **kwargs)

    if norm == "layer":
        return layer_normalization(name=name, **kwargs)

    raise ValueError(f"Normalization type '{norm}' not supported")
