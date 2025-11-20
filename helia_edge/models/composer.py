"""
# Composer Model API

This module provides utility functions to compose a sequential set of networks/layers.

Classes:
    ComposerLayerParams: Composer layer parameters
    ComposerParams: Composer Network parameters
    ComposerModel: Helper class to generate model from parameters

Functions:
    composer_layer: Composes a sequential set of networks/layers

"""

import logging

import keras
from pydantic import BaseModel, Field

from ..layers.normalization import batch_normalization
from ..layers.convolutional import conv2d
from ..layers.squeeze_excite import se_layer
from ..layers.activations import relu6
from .utils import load_model

logger = logging.getLogger(__name__)


class ComposerLayerParams(BaseModel):
    """Composer layer parameters

    Attributes:
        name (str): Layer name
        params (dict): Layer arguments
    """

    name: str = Field(..., description="Layer name")
    params: dict = Field(default_factory=dict, description="Layer arguments")


class ComposerParams(BaseModel):
    """Composer Network parameters

    Attributes:
        layers (list[ComposerLayerParams]): Network layers
        include_top (bool): Include top
        output_activation (str | None): Output activation
        name (str): Model name
    """

    layers: list[ComposerLayerParams] = Field(default_factory=list, description="Network layers")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="Composer", description="Model name")


def composer_layer(
    x: keras.KerasTensor,
    params: ComposerParams,
    num_classes: int | None = None,
) -> keras.KerasTensor:
    """Composes a sequential set of networks/layers.
    Useful for adding custom layers to a pre-trained model (e.g. foundation).

    Args:
        x (keras.KerasTensor): Model input
        params (ComposerParams): Model parameters
        num_classes (int | None): Number of classes

    Returns:
        keras.KerasTensor: Model output
    """
    y = x
    for layer in params.layers:
        match layer.name:
            case "conv2d":
                y = conv2d(y, **layer.params)
            case "dense":
                y = keras.layers.Dense(**layer.params)(y)
            case "relu6":
                y = relu6(y)
            case "batch_norm":
                y = batch_normalization(y)
            case "se_block":
                y = se_layer(y, **layer.params)
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
        if num_classes is not None:
            y = keras.layers.Dense(num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    return y


class ComposerModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: ComposerParams | dict, num_classes: int | None = None):
        """Create layer from parameters"""
        if isinstance(params, dict):
            params = ComposerParams(**params)
        return composer_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: ComposerParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = ComposerModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)
