"""Random image flips with one parameter set for spatially aligned leaves."""

import keras

from ...utils import helia_export
from .base_augmentation import BaseAugmentation2D


@helia_export(path="helia_edge.layers.preprocessing.RandomFlip2D")
class RandomFlip2D(BaseAugmentation2D):
    """Training-only horizontal (width) and vertical (height) image flips."""

    joint = True

    def __init__(self, horizontal: bool = True, vertical: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.horizontal = horizontal
        self.vertical = vertical

    def get_random_transformations(self, input_shape):
        shape = (input_shape[0], 1, 1, 1)
        return {
            name: keras.random.uniform(shape, seed=self.random_generator) <= 0.5
            for name, enabled in (("horizontal", self.horizontal), ("vertical", self.vertical))
            if enabled
        }

    def augment_samples(self, inputs):
        x = inputs[self.SAMPLES]
        for name, enabled, axis in (
            ("horizontal", self.horizontal, self.width_axis),
            ("vertical", self.vertical, self.height_axis),
        ):
            if enabled:
                x = keras.ops.where(inputs[self.TRANSFORMS][name], keras.ops.flip(x, axis=axis), x)
        return x

    def get_config(self):
        return {**super().get_config(), "horizontal": self.horizontal, "vertical": self.vertical}
