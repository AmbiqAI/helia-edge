from typing import Callable
import keras


def se_layer(
    ratio: int = 8,
    name: str | None = None,
    squeeze_activation: str | Callable = "relu6",
    excite_activation: str | Callable = "hard_sigmoid",
) -> keras.Layer:
    """Squeeze & excite functional layer

    Implements Squeeze and Excite block as in
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).

    Args:
        ratio (Expansion ratio, optional): Expansion ratio. Defaults to 8.
        name (str|None, optional): Block name. Defaults to None.
        squeeze_activation (str|Callable, optional): Squeeze activation. Defaults to "relu6".
        excite_activation (str|Callable, optional): Excite activation. Defaults to "hard_sigmoid".

    Returns:
        keras.Layer: Functional SE layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        num_chan = x.shape[-1]
        # Squeeze
        name_pool = f"{name}_pool" if name else None
        name_sq = f"{name}_sq" if name else None
        name_sq_act = f"{name}_sq.act" if name else None
        y = keras.layers.GlobalAveragePooling2D(name=name_pool, keepdims=True)(x)
        y = keras.layers.Conv2D(filters=int(num_chan // ratio), kernel_size=(1, 1), use_bias=True, name=name_sq)(y)
        y = keras.layers.Activation(squeeze_activation, name=name_sq_act)(y)
        # Excite
        name_ex = f"{name}_ex" if name else None
        name_ex_act = f"{name}_ex.act" if name else None
        name_ex_mul = f"{name}_ex.mul" if name else None
        y = keras.layers.Conv2D(num_chan, kernel_size=(1, 1), use_bias=True, name=name_ex)(y)
        y = keras.layers.Activation(excite_activation, name=name_ex_act)(y)
        y = keras.layers.Multiply(name=name_ex_mul)([x, y])
        return y

    # END DEF

    return layer
