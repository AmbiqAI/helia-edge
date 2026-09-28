"""Repeated batchwise random choices sharing the standard training contract."""

import keras
from .random_choice import RandomChoice
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomAugmentation1DPipeline")
class RandomAugmentation1DPipeline(RandomChoice):
    def __init__(
        self,
        layers: list[keras.Layer],
        augmentations_per_sample: int = 1,
        rate: float = 1.0,
        batchwise: bool = True,
        force_training: bool = False,
        **kwargs,
    ):
        super().__init__(layers=layers, batchwise=batchwise, **kwargs)
        if (
            isinstance(augmentations_per_sample, bool)
            or not isinstance(augmentations_per_sample, int)
            or augmentations_per_sample < 0
        ):
            raise ValueError("augmentations_per_sample must be a nonnegative integer")
        if not 0 <= rate <= 1:
            raise ValueError("rate must be in [0,1]")
        self.augmentations_per_sample = augmentations_per_sample
        self.rate = rate
        self.force_training = force_training

    def batch_augment(self, inputs, transformations=None):
        if transformations is not None:
            raise ValueError("Supply explicit transformations to child layers, not the random pipeline")
        for _ in range(self.augmentations_per_sample):
            if self.rate == 0:
                continue
            if self.rate == 1:
                inputs = super().batch_augment(inputs)
            else:
                apply = keras.random.uniform((), seed=self.random_generator) < self.rate
                current = inputs
                inputs = keras.ops.cond(
                    apply, lambda: super(RandomAugmentation1DPipeline, self).batch_augment(current), lambda: current
                )
        return inputs

    def call(self, inputs, training=None, transformations=None):
        return super().call(inputs, training=True if self.force_training else training, transformations=transformations)

    def get_config(self):
        return {
            **super().get_config(),
            "augmentations_per_sample": self.augmentations_per_sample,
            "rate": self.rate,
            "force_training": self.force_training,
        }


@helia_export(path="helia_edge.layers.preprocessing.RandomAugmentation2DPipeline")
class RandomAugmentation2DPipeline(RandomAugmentation1DPipeline):
    """The same batchwise composition contract for image transforms."""
