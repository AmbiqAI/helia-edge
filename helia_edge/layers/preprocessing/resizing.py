"""Deterministic resizing; aligned discrete leaves use nearest interpolation."""

import keras
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


def _resize(layer, x, interpolation):
    axis = 2 if layer.data_format == "channels_first" else 1
    if layer.NDIMS == 3:
        x = keras.ops.expand_dims(x, axis)
        size = (1, layer.duration)
    else:
        size = (layer.height, layer.width)
    x = keras.ops.image.resize(x, size, interpolation=interpolation, data_format=layer.data_format)
    return keras.ops.squeeze(x, axis) if layer.NDIMS == 3 else x


def _discrete(layer, inputs):
    x = inputs[layer.SAMPLES]
    axes = (
        ((layer.data_axis, layer.duration),)
        if layer.NDIMS == 3
        else ((layer.height_axis, layer.height), (layer.width_axis, layer.width))
    )
    for axis, size in axes:
        length = keras.ops.shape(x)[axis]
        # Half-pixel nearest indices preserve integer labels without float casts.
        indices = keras.ops.cast(
            keras.ops.floor((keras.ops.arange(size, dtype="float32") + 0.5) * length / size), "int32"
        )
        indices = keras.ops.minimum(indices, length - 1)
        x = keras.ops.take(x, indices, axis=axis)
    return x


@helia_export(path="helia_edge.layers.preprocessing.Resizing1D")
class Resizing1D(BaseAugmentation1D):
    training_only = False
    joint = True

    def __init__(self, duration: int, **kwargs):
        super().__init__(**kwargs)
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.duration = duration

    def augment_samples(self, inputs):
        return _resize(self, inputs[self.SAMPLES], "bicubic")

    def augment_targets(self, inputs):
        return _discrete(self, inputs)

    def get_config(self):
        return {**super().get_config(), "duration": self.duration}


@helia_export(path="helia_edge.layers.preprocessing.Resizing2D")
class Resizing2D(BaseAugmentation2D):
    training_only = False
    joint = True

    def __init__(self, height: int, width: int, interpolation: str = "bicubic", **kwargs):
        super().__init__(**kwargs)
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def augment_samples(self, inputs):
        return _resize(self, inputs[self.SAMPLES], self.interpolation)

    def augment_targets(self, inputs):
        return _discrete(self, inputs)

    def get_config(self):
        return {**super().get_config(), "height": self.height, "width": self.width, "interpolation": self.interpolation}
