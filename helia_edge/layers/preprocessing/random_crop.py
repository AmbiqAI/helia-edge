"""
# Random Crop Layers API

This module provides classes to build random crop layers.

Classes:
    RandomCrop1D: Random crop 1D
    RandomCrop2D: Random crop 2D

"""

import keras
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomCrop1D")
class RandomCrop1D(BaseAugmentation1D):
    duration: int
    unique_batch: bool

    def __init__(self, duration: int, unique_batch: bool = False, **kwargs):
        """Randomly crop 1D input samples.

        Args:
            duration (int): Duration of the output samples.
            unique_batch (bool): If True, each sample in the batch will have a unique crop.

        Example:

        ```python
            duration = 100
            lyr = RandomCrop1D(duration=duration)
            x = np.random.randn(32, 1000, 1)
            y = lyr(x, training=True)
        ```
        """

        super().__init__(**kwargs)
        self.duration = duration
        self.unique_batch = unique_batch

    def random_crop(self, sample, start):
        """Randomly crop single sample"""
        ch_size = keras.ops.shape(sample)[self.ch_axis]
        if self.data_format == "channels_first":
            return keras.ops.slice(sample, [0, start], [ch_size, self.duration])
        return keras.ops.slice(sample, [start, 0], [self.duration, ch_size])
        # END IF

    # END DEF

    def get_random_transformations(self, input_shape):
        """Generate random start indices for cropping."""
        batch_size = input_shape[0]
        duration_size = input_shape[self.data_axis]
        if duration_size < self.duration:
            raise ValueError(f"Input duration ({duration_size}) must be greater than output duration ({self.duration})")

        d_diff = duration_size - self.duration
        if self.unique_batch:
            start = keras.random.randint(
                shape=(batch_size,), minval=0, maxval=int(d_diff + 1), seed=self.random_generator, dtype="int32"
            )
        else:
            start = keras.random.randint(
                shape=(), minval=0, maxval=int(d_diff + 1), seed=self.random_generator, dtype="int32"
            )
            start = keras.ops.broadcast_to(start, [batch_size])
        return {"start": start}

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Augment single sample with random crop."""
        sample = inputs[self.SAMPLES]
        start = inputs[self.TRANSFORMS]["start"]

        if self.training:
            sample = self.random_crop(sample, start)
        return sample

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)
        output_shape[self.data_axis] = self.duration
        return tuple(input_shape)

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "duration": self.duration,
                "unique_batch": self.unique_batch,
            }
        )
        return config


@helia_export(path="helia_edge.layers.preprocessing.RandomCrop2D")
class RandomCrop2D(BaseAugmentation2D):
    height: int
    width: int
    unique_batch: bool

    def __init__(self, height: int, width: int, unique_batch: bool = False, **kwargs):
        """Randomly crop 2D input samples.

        Args:
            height (int): Height of the output samples.
            width (int): Width of the output samples.
            unique_batch (bool): If True, each sample in the batch will have a unique crop.

        Example:

        ```python
            height = 32
            width = 32
            lyr = RandomCrop2D(height=height, width=width)
            x = np.random.randn(32, 64, 64, 3)
            y = lyr(x, training=True)
        ```
        """

        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.unique_batch = unique_batch

    def random_crop(self, sample, start_h, start_w):
        """Randomly crop single sample"""
        if self.data_format == "channels_first":
            return keras.ops.slice(sample, [0, start_h, start_w], [-1, self.height, self.width])
        return keras.ops.slice(sample, [start_h, start_w, 0], [self.height, self.width, -1])
        # END IF

    def get_random_transformations(self, input_shape):
        """Generate random start indices for cropping."""
        batch_size = input_shape[0]
        height_size = input_shape[self.height_axis]
        width_size = input_shape[self.width_axis]
        if height_size < self.height:
            raise ValueError(f"Input height ({height_size}) must be greater than output height ({self.height})")
        if width_size < self.width:
            raise ValueError(f"Input width ({width_size}) must be greater than output width ({self.width})")

        h_diff = height_size - self.height
        w_diff = width_size - self.width
        if self.unique_batch:
            start_h = keras.random.randint(
                shape=(batch_size,), minval=0, maxval=int(h_diff + 1), seed=self.random_generator, dtype="int32"
            )
            start_w = keras.random.randint(
                shape=(batch_size,), minval=0, maxval=int(w_diff + 1), seed=self.random_generator, dtype="int32"
            )
        else:
            start_h = keras.random.randint(
                shape=(), minval=0, maxval=int(h_diff + 1), seed=self.random_generator, dtype="int32"
            )
            start_h = keras.ops.broadcast_to(start_h, [batch_size])
            start_w = keras.random.randint(
                shape=(), minval=0, maxval=int(w_diff + 1), seed=self.random_generator, dtype="int32"
            )
            start_w = keras.ops.broadcast_to(start_w, [batch_size])
        return {"start_h": start_h, "start_w": start_w}

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Augment single sample with random crop."""
        sample = inputs[self.SAMPLES]
        start_h = inputs[self.TRANSFORMS]["start_h"]
        start_w = inputs[self.TRANSFORMS]["start_w"]

        if self.training:
            sample = self.random_crop(sample, start_h, start_w)
        return sample

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute the output shape of the layer."""
        output_shape = list(input_shape)
        output_shape[self.height_axis] = self.height
        output_shape[self.width_axis] = self.width
        return tuple(input_shape)

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "unique_batch": self.unique_batch,
            }
        )
        return config
