"""
# SpecAugment Layer API

This module provides classes to build SpecAugment layers.

Classes:
    SpecAugment2D: SpecAugment 2D data
"""

import keras

from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.SpecAugment2D")
class SpecAugment2D(keras.Layer):
    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        n_freq_mask: int = 1,
        n_time_mask: int = 1,
        mask_value: float = 0.0,
        **kwargs,
    ):
        """SpecAugment layer w/o time warping

        Args:
            freq_mask_param (int): Frequency Mask Parameter (F in the paper)
            time_mask_param (int): Time Mask Parameter (T in the paper)
            n_freq_mask (int, optional): Number of frequency masks to apply (mF in the paper). Defaults to 1.
            n_time_mask (int, optional): Number of time masks to apply (mT in the paper). Defaults to 1.
            mask_value (float, optional): Imputation value. Defaults to zero.

        """
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_mask = n_freq_mask
        self.n_time_mask = n_time_mask
        self.mask_value = keras.ops.cast(mask_value, "float32")

    def _frequency_mask_single(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Generate the frequency mask for a single spectrogram

        Args:
            x (keras.KerasTensor): The input mel spectrogram

        Returns:
            keras.KerasTensor: Frequency mask
        """
        n_mels = keras.ops.shape(x)[1]
        freq_indices = keras.ops.reshape(keras.ops.arange(n_mels, dtype="int32"), (1, -1, 1))

        # We use the paper's notation
        f = keras.random.randint(shape=(), minval=0, maxval=self.freq_mask_param, dtype="int32")
        f0 = keras.random.randint(shape=(), minval=0, maxval=n_mels - f, dtype="int32")

        condition = keras.ops.logical_and(freq_indices >= f0, freq_indices <= f0 + f)
        return keras.ops.cast(condition, dtype="float32")

    def _frequency_masks(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Generate the frequency masks

        Args:
            x (keras.KerasTensor): The input mel spectrogram

        Returns:
            keras.KerasTensor: The mel spectrogram with the frequency masks applied
        """
        mel_repeated = keras.ops.repeat(keras.ops.expand_dims(x, 0), self.n_freq_mask, axis=0)
        masks = keras.ops.cast(keras.ops.map(f=self._frequency_mask_single, xs=mel_repeated), dtype="bool")
        mask = keras.ops.any(masks, 0)
        return keras.ops.where(mask, self.mask_value, x)

    def _time_mask_single(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Generate the time mask for a single spectrogram

        Args:
            x (keras.KerasTensor): The input mel spectrogram

        Returns:
            keras.KerasTensor: Time mask

        """
        n_steps = keras.ops.shape(x)[0]
        time_indices = keras.ops.reshape(keras.ops.arange(n_steps, dtype="int32"), (-1, 1, 1))

        # We use the paper's notation
        t = keras.random.randint(shape=(), minval=0, maxval=self.time_mask_param, dtype="int32")
        t0 = keras.random.randint(shape=(), minval=0, maxval=n_steps - t, dtype="int32")

        condition = keras.ops.logical_and(time_indices >= t0, time_indices <= t0 + t)
        return keras.ops.cast(condition, dtype="float32")

    def _time_masks(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Generate the time masks

        Args:
            x (keras.KerasTensor): The input mel spectrogram

        Returns:
            keras.KerasTensor: The mel spectrogram with the time masks applied
        """
        mel_repeated = keras.ops.repeat(keras.ops.expand_dims(x, 0), self.n_time_mask, axis=0)
        masks = keras.ops.cast(keras.ops.map(f=self._time_mask_single, xs=mel_repeated), dtype="bool")
        mask = keras.ops.any(masks, 0)
        return keras.ops.where(mask, self.mask_value, x)

    def _apply_spec_augment(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Apply SpecAugment to a single mel spectrogram

        Args:
            x (keras.KerasTensor): The input mel spectrogram

        Returns:
            keras.KerasTensor: The mel spectrogram with the SpecAugment applied
        """
        if self.n_freq_mask >= 1:
            x = self._frequency_masks(x)
        if self.n_time_mask >= 1:
            x = self._time_masks(x)
        return x

    def call(self, inputs: keras.KerasTensor, training=None, **kwargs) -> keras.KerasTensor:
        """Applies the SpecAugment operation to the input Mel Spectrogram

        Args:
            inputs (keras.KerasTensor): The input mel spectrogram
            training (bool, optional): Whether the model is training. Defaults to None.

        Returns:
            keras.KerasTensor: The mel spectrogram with the SpecAugment applied
        """

        if training:
            inputs_masked = keras.ops.map(f=self._apply_spec_augment, xs=inputs)
            return inputs_masked
        return inputs

    def get_config(self):
        """Configuration to initialize the layer"""
        config = {
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "n_freq_mask": self.n_freq_mask,
            "n_time_mask": self.n_time_mask,
            "mask_value": self.mask_value.numpy(),
        }
        return config
