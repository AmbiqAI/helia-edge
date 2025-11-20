"""
# Random Cutout Layer API

This module provides classes to build random cutout layers.

Classes:
    RandomCutout1D: Random cutout 1D
    RandomCutout2D: Random cutout 2D

"""

import keras

from ...utils import parse_factor
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import helia_export


@helia_export(path="helia_edge.layers.preprocessing.RandomCutout1D")
class RandomCutout1D(BaseAugmentation1D):
    cutouts: int
    factor: tuple[float, float]
    fill_mode: str
    fill_value: float

    def __init__(
        self,
        factor: float | tuple[float, float] = 0.1,
        cutouts: int = 1,
        fill_mode="constant",
        fill_value: float = 0.0,
        **kwargs,
    ):
        """Apply random cutout to the input. This is similar to its 2D counterpart where a random portion of the input is cutout.
        We allow providing a range for the factor and cutouts to randomly pick the values.

        Args:
            factor (float|tuple[float,float]): Factor of the duration to cutout. If tuple, factor is randomly picked between the values.
            cutouts (int): Number of cutouts to apply.
            fill_mode (str): Fill mode. "constant" or "normal".
            fill_value (float): Fill value for the cutout.

        """
        super().__init__(**kwargs)

        self.factor = parse_factor(factor, min_value=0, max_value=1, param_name="factor")
        self.cutouts = cutouts
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        if fill_mode not in ["normal", "constant"]:
            raise ValueError(f'`fill_mode` should be "normal" or "constant". Got `fill_mode`={fill_mode}')
        # END IF

    def call(self, inputs, training: bool = True):
        """Override the call method to apply multiple cutouts."""
        self.training = training
        if not self.training:
            return inputs
        inputs, metadata = self._format_inputs(inputs)
        outputs = keras.ops.fori_loop(
            lower=0,
            upper=self.cutouts,
            body_fun=lambda _, x: self.batch_augment(x),
            init_val=self.batch_augment(inputs),
        )
        return self._format_outputs(outputs, metadata)

    def get_random_transformations(self, input_shape):
        """Generate random cutout locations, sizes, and fill values."""
        batch_size = input_shape[0]
        duration_size = input_shape[self.data_axis]

        cut_size = keras.random.randint(
            shape=(batch_size,),
            minval=int(duration_size * self.factor[0]),
            maxval=int(duration_size * self.factor[1]) + 1,
            dtype="int32",
            seed=self.random_generator,
        )
        cut_start = keras.random.randint(
            shape=(batch_size,),
            minval=0,
            maxval=int(duration_size * (1 - self.factor[1]) + 1),
            dtype="int32",
            seed=self.random_generator,
        )
        if self.fill_mode == "constant":
            fill = keras.ops.ones(input_shape) * self.fill_value
        else:
            fill = keras.random.normal(input_shape, mean=0, stddev=self.fill_value, seed=self.random_generator)

        return {
            "cut_start": cut_start,
            "cut_size": cut_size,
            "fill": fill,
        }

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Apply cutout to the input."""
        sample = inputs[self.SAMPLES]
        transforms = inputs[self.TRANSFORMS]
        cut_start = transforms["cut_start"]
        cut_size = transforms["cut_size"]
        fill = transforms["fill"]

        duration_size = sample.shape[self.data_axis]
        ch_size = sample.shape[self.ch_axis]

        if self.data_format == "channels_first":
            reshape_size = (1, duration_size)
            tile_size = (ch_size, 1)
        else:
            reshape_size = (duration_size, 1)
            tile_size = (1, ch_size)

        mask = keras.ops.tile(
            keras.ops.reshape(
                keras.ops.logical_and(
                    keras.ops.arange(duration_size) >= cut_start, keras.ops.arange(duration_size) < cut_start + cut_size
                ),
                reshape_size,
            ),
            tile_size,
        )
        result = keras.ops.where(mask, fill, sample)
        return result

    def get_config(self):
        """Serialize the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "cutouts": self.cutouts,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
            }
        )
        return config


@helia_export(path="helia_edge.layers.preprocessing.RandomCutout2D")
class RandomCutout2D(BaseAugmentation2D):
    cutouts: int
    factor: tuple[float, float]
    fill_mode: str
    fill_value: float

    def __init__(
        self,
        factor: float | tuple[float, float] = 0.1,
        cutouts: int = 1,
        fill_mode="constant",
        fill_value: float = 0.0,
        **kwargs,
    ):
        """Apply random cutout to the input. This is similar to its 1D counterpart where a random portion of the input is cutout.
        We allow providing a range for the factor and cutouts to randomly pick the values.

        Args:
            factor (float|tuple[float,float]): Factor of the dimensions to cutout. If tuple, factor is randomly picked between the values.
            cutouts (int): Number of cutouts to apply.
            fill_mode (str): Fill mode. "constant" or "normal".
            fill_value (float): Fill value for the cutout.

        """
        super().__init__(**kwargs)

        self.factor = parse_factor(factor, min_value=0, max_value=1, param_name="factor")
        self.cutouts = cutouts
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        if fill_mode not in ["normal", "constant"]:
            raise ValueError(f'`fill_mode` should be "normal" or "constant". Got `fill_mode`={fill_mode}')
        # END IF

    def call(self, inputs, training: bool = True):
        """Override the call method to apply multiple cutouts."""
        self.training = training
        if not self.training:
            return inputs
        inputs, metadata = self._format_inputs(inputs)
        outputs = keras.ops.fori_loop(
            lower=0,
            upper=self.cutouts,
            body_fun=lambda _, x: self.batch_augment(x),
            init_val=self.batch_augment(inputs),
        )
        return self._format_outputs(outputs, metadata)

    def get_random_transformations(self, input_shape):
        """Generate random cutout locations, sizes, and fill values."""
        batch_size = input_shape[0]
        height_size = input_shape[self.height_axis]
        width_size = input_shape[self.width_axis]

        cut_height = keras.random.randint(
            shape=(batch_size,),
            minval=int(height_size * self.factor[0]),
            maxval=int(height_size * self.factor[1]) + 1,
            dtype="int32",
            seed=self.random_generator,
        )
        cut_width = keras.random.randint(
            shape=(batch_size,),
            minval=int(width_size * self.factor[0]),
            maxval=int(width_size * self.factor[1]) + 1,
            dtype="int32",
            seed=self.random_generator,
        )
        cut_start_height = keras.random.randint(
            shape=(batch_size,),
            minval=0,
            maxval=int(height_size * (1 - self.factor[1]) + 1),
            dtype="int32",
            seed=self.random_generator,
        )
        cut_start_width = keras.random.randint(
            shape=(batch_size,),
            minval=0,
            maxval=int(width_size * (1 - self.factor[1]) + 1),
            dtype="int32",
            seed=self.random_generator,
        )
        if self.fill_mode == "constant":
            fill = keras.ops.ones(input_shape) * self.fill_value
        else:
            fill = keras.random.normal(input_shape, mean=0, stddev=self.fill_value, seed=self.random_generator)

        return {
            "cut_start_height": cut_start_height,
            "cut_start_width": cut_start_width,
            "cut_height": cut_height,
            "cut_width": cut_width,
            "fill": fill,
        }

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Apply cutout to the input."""
        sample = inputs[self.SAMPLES]
        transforms = inputs[self.TRANSFORMS]
        cut_start_height = transforms["cut_start_height"]
        cut_start_width = transforms["cut_start_width"]
        cut_height = transforms["cut_height"]
        cut_width = transforms["cut_width"]
        fill = transforms["fill"]

        height_size = sample.shape[self.height_axis]
        width_size = sample.shape[self.width_axis]
        ch_size = sample.shape[self.ch_axis]

        if self.data_format == "channels_first":
            reshape_size = (1, height_size, width_size)
            tile_size = (ch_size, 1, 1)
        else:
            reshape_size = (height_size, width_size, 1)
            tile_size = (1, 1, ch_size)

        mask = keras.ops.tile(
            keras.ops.reshape(
                keras.ops.logical_and(
                    keras.ops.logical_and(
                        keras.ops.arange(height_size) >= cut_start_height,
                        keras.ops.arange(height_size) < cut_start_height + cut_height,
                    ),
                    keras.ops.logical_and(
                        keras.ops.arange(width_size) >= cut_start_width,
                        keras.ops.arange(width_size) < cut_start_width + cut_width,
                    ),
                ),
                reshape_size,
            ),
            tile_size,
        )
        result = keras.ops.where(mask, fill, sample)
        return result

    def get_config(self):
        """Serialize the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "cutouts": self.cutouts,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
            }
        )
        return config
