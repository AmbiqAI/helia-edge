"""
# Random Channel Layer API

This module provides classes to build random channel layers.

Classes:
    RandomChannel: Random channel

"""

import keras

from .base_augmentation import BaseAugmentation
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomChannel")
class RandomChannel(BaseAugmentation):
    batchwise: bool

    def __init__(self, batchwise: bool = False, **kwargs):
        """Randomly picks a single channel from the input samples.

        Args:
            batchwise (bool): If True, grabs same channel from all samples in batch.

        """
        super().__init__(**kwargs)
        self.batchwise = batchwise

    def get_random_transformations(self, input_shape):
        """Generate random channel index."""

        batch = input_shape[0]
        if self.data_format == "channels_first":
            ch_size = input_shape[self.ch_axis]
        else:
            ch_size = input_shape[self.ch_axis]

        # Pick same channel for all samples in the batch
        if self.batchwise:
            ch_idx = keras.random.randint(shape=(), minval=0, maxval=ch_size, seed=self.random_generator, dtype="int32")
            ch_idx = keras.ops.broadcast_to(ch_idx, [batch])
        else:
            ch_idx = keras.random.randint(
                shape=(batch,), minval=0, maxval=ch_size, seed=self.random_generator, dtype="int32"
            )
        return {"channel": ch_idx}

    def augment_samples(self, inputs):
        """Augment samples"""
        # If batchwise, easier to grab the channel here
        if self.batchwise:
            samples = inputs[self.SAMPLES]
            channel = inputs[self.TRANSFORMS]["channel"]
            channel = keras.ops.take(channel, 0, axis=0)
            return keras.ops.expand_dims(keras.ops.take(samples, channel, axis=self.ch_axis), axis=self.ch_axis)
        # Otherwise let the augment_sample method handle it
        else:
            super().augment_samples(inputs)

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Augment a sample during training."""
        sample = inputs[self.SAMPLES]
        channel = inputs[self.TRANSFORMS]["channel"]
        keras.ops.expand_dims(keras.ops.take(sample, channel, axis=self.ch_axis), axis=self.ch_axis)
