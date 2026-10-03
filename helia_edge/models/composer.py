"""
# Composer Model API

This module provides utility functions to compose a sequential set of networks/layers.

Parameters are in ``helia_edge.models.composer_params``.

Functions:
    build: Composer model from ``ComposerParams``
    composer_layer: Composes a sequential set of networks/layers

"""

import logging

import keras

from ..layers.activations import relu6
from ..layers.convolutional import conv2d
from ..layers.normalization import batch_normalization
from ..layers.squeeze_excite import se_layer
from .composer_params import ComposerParams
from .utils import load_model

logger = logging.getLogger(__name__)


def composer_layer(x: keras.KerasTensor, params: ComposerParams) -> keras.KerasTensor:
    """Composes a sequential set of networks/layers.
    Useful for adding custom layers to a pre-trained model (e.g. foundation).

    Args:
        x (keras.KerasTensor): Model input
        params (ComposerParams): Model parameters

    Returns:
        keras.KerasTensor: Model output
    """
    y = x
    for layer in params.layers:
        match layer.name:
            case "conv2d":
                y = conv2d(**layer.params)(y)
            case "dense":
                y = keras.layers.Dense(**layer.params)(y)
            case "relu6":
                y = relu6(**layer.params)(y)
            case "batch_norm":
                y = batch_normalization(**layer.params)(y)
            case "se_block":
                y = se_layer(**layer.params)(y)
            case "load_model":
                prev_model = load_model(layer.params["model_file"])
                trainable = layer.params.get("trainable", True)
                if not trainable:
                    logger.info(f"Freezing model {prev_model.name}")
                prev_model.trainable = trainable
                y = prev_model(y, training=trainable)
            case _:
                raise ValueError(f"Unknown layer {layer.name}")
        # END MATCH
    # END FOR

    if params.include_top:
        if params.num_classes is not None:
            y = keras.layers.Dense(params.num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    return y


def build(
    params: ComposerParams, input_shape: tuple[int, ...], *, batch_size: int | None = None, name: str | None = None
) -> keras.Model:
    """Build a Composer model.

    Args:
        params (ComposerParams): Model parameters.
        input_shape (tuple[int, ...]): Input shape without the batch axis.
        batch_size (int | None): Static batch size; None for a dynamic batch.
        name (str | None): Model name; the family when None.

    Returns:
        keras.Model: The model, named ``composer`` unless ``name`` is given.
    """
    inputs = keras.Input(shape=input_shape, batch_size=batch_size, name="inputs")
    return keras.Model(inputs=inputs, outputs=composer_layer(inputs, params), name=name or params.family)
