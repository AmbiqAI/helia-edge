"""
# Activation Layers API

Functions:
    glu: Gated linear unit layer
    relu: ReLU activation layer
    swish: Swish activation layer
    relu6: Hard ReLU activation layer
    mish: Mish activation layer
    gelu: GeLU activation layer
    sigmoid: Sigmoid activation layer
    hard_sigmoid: Hard sigmoid activation layer

Please check [Keras Activations](https://keras.io/api/layers/activations/) for additional activations.
"""

import keras


def glu(dim: int = -1, hard: bool = False, name: str | None = None) -> keras.Layer:
    """Gated linear unit layer

    Args:
        dim (int, optional): Dimension to split. Defaults to -1.
        hard (bool, optional): Use hard sigmoid. Defaults to False.
        name (str|None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional GLU layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        out, gate = keras.ops.split(x, indices_or_sections=2, axis=dim)
        act = keras.activations.sigmoid if hard else keras.activations.hard_sigmoid
        gate = keras.layers.Activation(act)(gate)
        x = keras.layers.Multiply()([out, gate])
        return x

    # END DEF

    return layer


def relu(name: str | None = None, truncated: bool = False, **kwargs) -> keras.Layer:
    """ReLU activation layer w/ optional truncation to ReLU6

    Args:
        name (str|None, optional): Layer name. Defaults to None.
        truncated (bool, optional): Truncate to ReLU6. Defaults to False.

    Returns:
        keras.Layer: Functional ReLU layer

    """
    name = name + "_act" if name else None
    act = keras.activations.relu6 if truncated else keras.activations.relu
    return keras.layers.Activation(activation=act, name=name, **kwargs)


def swish(name: str | None = None, hard: bool = False, **kwargs) -> keras.Layer:
    """Swish activation layer w/ optional hard variant

    Args:
        name (str|None, optional): Layer name. Defaults to None.
        hard (bool, optional): Use hard swish. Defaults to False.

    Returns:
        keras.Layer: Functional Swish layer
    """
    name = name + "_act" if name else None
    act = keras.activations.hard_swish if hard else keras.activations.swish
    return keras.layers.Activation(act, name=name)


def relu6(name: str | None = None, **kwargs) -> keras.Layer:
    """Hard ReLU activation layer

    Args:
        name (str|None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional ReLU6 layer
    """
    name = name + "_act" if name else None
    return keras.layers.Activation("relu6", name=name, **kwargs)


def mish(name: str | None = None, **kwargs) -> keras.Layer:
    """Mish activation layer

    Args:
        name (str|None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional Mish layer
    """
    name = name + "_act" if name else None
    return keras.layers.Activation(keras.activations.mish, name=name, **kwargs)


def gelu(name: str | None = None, **kwargs) -> keras.Layer:
    """GeLU activation layer

    Args:
        name (str|None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional GeLU layer
    """
    name = name + "_act" if name else None
    return keras.layers.Activation(keras.activations.gelu, name=name, **kwargs)


def sigmoid(name: str | None = None, hard: bool = False, **kwargs) -> keras.Layer:
    """Sigmoid activation layer

    Args:
        name (str|None, optional): Layer name. Defaults to None.
        hard (bool, optional): Use hard sigmoid. Defaults to False.

    Returns:
        keras.Layer: Functional Sigmoid layer
    """
    name = name + "_act" if name else None
    activation = keras.activations.hard_sigmoid if hard else keras.activations.sigmoid
    return keras.layers.Activation(activation, name=name)


def hard_sigmoid(name: str | None = None, **kwargs) -> keras.Layer:
    """Hard sigmoid activation layer

    Args:
        name (str|None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional Hard sigmoid layer
    """
    name = name + "_act" if name else None
    return keras.layers.Activation(keras.activations.hard_sigmoid, name=name)
