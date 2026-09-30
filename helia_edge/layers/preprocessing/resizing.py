"""Deterministic resizing; explicit target interpolation and discrete masks."""

import keras

from ...utils import helia_export
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D


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


def _target_policy(value):
    if value not in (None, "nearest", "signal"):
        raise ValueError("target_interpolation must be None, 'nearest', or 'signal'")
    return value


def _targets(layer, inputs):
    if layer.target_interpolation is None:
        raise ValueError("Set target_interpolation explicitly for selected aligned targets")
    if layer.target_interpolation == "nearest":
        return _discrete(layer, inputs)
    # Continuous targets follow signal precision, including integer input conversion.
    values = keras.ops.cast(inputs[layer.SAMPLES], layer.compute_dtype)
    return layer.augment_samples({**inputs, layer.SAMPLES: values})


@helia_export(path="helia_edge.layers.preprocessing.Resizing1D")
class Resizing1D(BaseAugmentation1D):
    """Resize signals with bicubic interpolation during training and inference.

    Selected masks use nearest-neighbor indices and retain their dtype.
    Selected targets require an explicit interpolation policy.

    Args:
        duration: Positive output length in samples.
        target_interpolation: "nearest" preserves discrete target values;
            "signal" uses the signal interpolation and compute dtype. None
            rejects resizing selected aligned targets.
        **kwargs (Any): Base augmentation options, including data_format,
            aligned_targets and aligned_masks.

    Raises:
        ValueError: An output dimension is nonpositive, the target policy is
            invalid, or selected targets have no interpolation policy.
    """

    training_only = False
    joint = True

    def __init__(self, duration: int, target_interpolation: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.duration = duration
        self.target_interpolation = _target_policy(target_interpolation)

    def augment_samples(self, inputs):
        return _resize(self, inputs[self.SAMPLES], "bicubic")

    def augment_targets(self, inputs):
        return _targets(self, inputs)

    def augment_masks(self, inputs):
        return _discrete(self, inputs)

    def get_config(self):
        return {**super().get_config(), "duration": self.duration, "target_interpolation": self.target_interpolation}


@helia_export(path="helia_edge.layers.preprocessing.Resizing2D")
class Resizing2D(BaseAugmentation2D):
    """Resize images and aligned data during training and inference.

    Selected masks use nearest-neighbor indices and retain their dtype.
    Selected targets require an explicit interpolation policy.

    Args:
        height: Positive output height in pixels.
        width: Positive output width in pixels.
        interpolation: Signal interpolation method accepted by Keras image.resize.
        target_interpolation: "nearest" preserves discrete target values;
            "signal" uses the signal interpolation and compute dtype. None
            rejects resizing selected aligned targets.
        **kwargs (Any): Base augmentation options, including data_format,
            aligned_targets and aligned_masks.

    Raises:
        ValueError: An output dimension is nonpositive, the target policy is
            invalid, or selected targets have no interpolation policy.
    """

    training_only = False
    joint = True

    def __init__(
        self, height: int, width: int, interpolation: str = "bicubic", target_interpolation: str | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.target_interpolation = _target_policy(target_interpolation)

    def augment_samples(self, inputs):
        return _resize(self, inputs[self.SAMPLES], self.interpolation)

    def augment_targets(self, inputs):
        return _targets(self, inputs)

    def augment_masks(self, inputs):
        return _discrete(self, inputs)

    def get_config(self):
        return {
            **super().get_config(),
            "height": self.height,
            "width": self.width,
            "interpolation": self.interpolation,
            "target_interpolation": self.target_interpolation,
        }
