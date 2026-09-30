"""Signal-only spectrogram masking using the shared augmentation contract."""

import keras

from ...utils import helia_export
from .base_augmentation import BaseAugmentation2D
from .random_cutout import interval_mask


@helia_export(path="helia_edge.layers.preprocessing.SpecAugment2D")
class SpecAugment2D(BaseAugmentation2D):
    """Mask frequency (height) and time (width), independently per example."""

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        n_freq_mask: int = 1,
        n_time_mask: int = 1,
        mask_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for value in (freq_mask_param, time_mask_param, n_freq_mask, n_time_mask):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Mask extents and counts must be nonnegative integers")
        self.freq_mask_param, self.time_mask_param = freq_mask_param, time_mask_param
        self.n_freq_mask, self.n_time_mask, self.mask_value = n_freq_mask, n_time_mask, mask_value

    def get_random_transformations(self, input_shape):
        mask = keras.ops.zeros(input_shape, dtype="bool")
        for axis, width, count in (
            (self.height_axis, self.freq_mask_param, self.n_freq_mask),
            (self.width_axis, self.time_mask_param, self.n_time_mask),
        ):
            if width > input_shape[axis]:
                raise ValueError("Mask maximum exceeds the input extent")
            for _ in range(count):
                mask = mask | interval_mask(self, input_shape, axis, 0, width)
        return {"mask": mask}

    def augment_samples(self, inputs):
        return keras.ops.where(
            inputs[self.TRANSFORMS]["mask"],
            keras.ops.cast(self.mask_value, inputs[self.SAMPLES].dtype),
            inputs[self.SAMPLES],
        )

    def get_config(self):
        return {
            **super().get_config(),
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "n_freq_mask": self.n_freq_mask,
            "n_time_mask": self.n_time_mask,
            "mask_value": self.mask_value,
        }
