"""Random crops sharing sampled offsets across spatially aligned leaves."""

import keras
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


def _starts(layer, batch, size, length):
    if not isinstance(size, int) or size < length:
        raise ValueError("Crop requires a static input extent at least as large as the output")
    value = keras.random.randint(
        (batch,) if layer.unique_batch else (), 0, size - length + 1, seed=layer.random_generator, dtype="int32"
    )
    return value if layer.unique_batch else keras.ops.broadcast_to(value, (batch,))


@helia_export(path="helia_edge.layers.preprocessing.RandomCrop1D")
class RandomCrop1D(BaseAugmentation1D):
    """Training-only crop; inference leaves the original duration unchanged."""

    joint = True

    def __init__(self, duration: int, unique_batch: bool = False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise ValueError("duration must be a positive integer")
        self.duration = duration
        self.unique_batch = unique_batch

    def get_random_transformations(self, input_shape):
        return {"start": _starts(self, input_shape[0], input_shape[self.data_axis], self.duration)}

    def augment_samples(self, inputs):
        x = inputs[self.SAMPLES]
        if self.data_format == "channels_first":
            x = keras.ops.transpose(x, (0, 2, 1))
        indices = inputs[self.TRANSFORMS]["start"][:, None, None] + keras.ops.arange(self.duration)[None, :, None]
        x = keras.ops.take_along_axis(x, indices, axis=1)
        return keras.ops.transpose(x, (0, 2, 1)) if self.data_format == "channels_first" else x

    def get_config(self):
        return {**super().get_config(), "duration": self.duration, "unique_batch": self.unique_batch}


@helia_export(path="helia_edge.layers.preprocessing.RandomCrop2D")
class RandomCrop2D(BaseAugmentation2D):
    """Training-only image crop; inference leaves the original extent unchanged."""

    joint = True

    def __init__(self, height: int, width: int, unique_batch: bool = False, **kwargs):
        super().__init__(**kwargs)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (height, width)):
            raise ValueError("height and width must be positive integers")
        self.height = height
        self.width = width
        self.unique_batch = unique_batch

    def get_random_transformations(self, input_shape):
        return {
            "start_h": _starts(self, input_shape[0], input_shape[self.height_axis], self.height),
            "start_w": _starts(self, input_shape[0], input_shape[self.width_axis], self.width),
        }

    def augment_samples(self, inputs):
        x = inputs[self.SAMPLES]
        if self.data_format == "channels_first":
            x = keras.ops.transpose(x, (0, 2, 3, 1))
        params = inputs[self.TRANSFORMS]
        h = params["start_h"][:, None, None, None] + keras.ops.arange(self.height)[None, :, None, None]
        w = params["start_w"][:, None, None, None] + keras.ops.arange(self.width)[None, None, :, None]
        x = keras.ops.take_along_axis(keras.ops.take_along_axis(x, h, axis=1), w, axis=2)
        return keras.ops.transpose(x, (0, 3, 1, 2)) if self.data_format == "channels_first" else x

    def get_config(self):
        return {**super().get_config(), "height": self.height, "width": self.width, "unique_batch": self.unique_batch}
