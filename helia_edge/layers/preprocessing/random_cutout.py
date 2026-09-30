"""Signal-only occlusion with explicit half-open intervals and owned RNG."""

import keras

from ...utils import helia_export, parse_factor
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D


def interval_mask(layer, shape, axis, minimum, maximum):
    """Sample one interval per example; returned mask broadcasts over channels."""
    length = shape[axis]
    width = keras.random.randint((shape[0],), minimum, maximum + 1, seed=layer.random_generator, dtype="int32")
    start = keras.ops.cast(
        keras.random.uniform((shape[0],), seed=layer.random_generator) * keras.ops.cast(length - width + 1, "float32"),
        "int32",
    )
    view = [shape[0]] + [1] * (len(shape) - 1)
    indices_view = [1] * len(shape)
    indices_view[axis] = length
    indices = keras.ops.reshape(keras.ops.arange(length), indices_view)
    start = keras.ops.reshape(start, view)
    width = keras.ops.reshape(width, view)
    return (indices >= start) & (indices < start + width)


def _parameters(layer, shape, axes):
    if layer.cutouts == 0:
        return None
    mask = keras.ops.zeros(shape, dtype="bool")
    for _ in range(layer.cutouts):
        region = keras.ops.ones(shape, dtype="bool")
        for axis in axes:
            length = shape[axis]
            region = region & interval_mask(
                layer, shape, axis, int(length * layer.factor[0]), int(length * layer.factor[1])
            )
        mask = mask | region
    fill = (
        keras.ops.full(shape, layer.fill_value, dtype=layer.compute_dtype)
        if layer.fill_mode == "constant"
        else keras.random.normal(shape, stddev=layer.fill_value, seed=layer.random_generator, dtype=layer.compute_dtype)
    )
    return {"mask": mask, "fill": fill}


def _initialize(layer, factor, cutouts, fill_mode, fill_value):
    layer.factor = parse_factor(factor, min_value=0, max_value=1, param_name="factor")
    if isinstance(cutouts, bool) or not isinstance(cutouts, int) or cutouts < 0:
        raise ValueError("cutouts must be a nonnegative integer")
    if fill_mode not in ("constant", "normal") or (fill_mode == "normal" and fill_value < 0):
        raise ValueError("fill_mode must be constant or normal with nonnegative standard deviation")
    layer.cutouts, layer.fill_mode, layer.fill_value = cutouts, fill_mode, fill_value


def _config(layer):
    return {
        "factor": layer.factor,
        "cutouts": layer.cutouts,
        "fill_mode": layer.fill_mode,
        "fill_value": layer.fill_value,
    }


@helia_export(path="helia_edge.layers.preprocessing.RandomCutout1D")
class RandomCutout1D(BaseAugmentation1D):
    """Replace random temporal intervals in each example during training.

    Regions share their positions across channels. Targets and masks remain
    unchanged; only signal values are occluded. Overlapping regions are allowed.

    Args:
        factor (float | tuple[float, float]): Fractional size bounds in [0, 1] for each spatial axis. A scalar
            sets the upper bound with a zero lower bound; a pair sets both bounds.
        cutouts (int): Nonnegative number of regions sampled per example.
        fill_mode (str): "constant" for a fixed value or "normal" for Gaussian noise.
        fill_value (float): Constant fill value, or the nonnegative standard deviation
            of zero-mean Gaussian noise when fill_mode is "normal".
        **kwargs (Any): Base augmentation options, including seed and data_format.

    Raises:
        ValueError: Size bounds, region count, fill mode or noise deviation
            violate these constraints.
    """

    def __init__(self, factor=0.1, cutouts=1, fill_mode="constant", fill_value=0.0, **kwargs):
        super().__init__(**kwargs)
        _initialize(self, factor, cutouts, fill_mode, fill_value)

    def get_random_transformations(self, input_shape):
        return _parameters(self, input_shape, (self.data_axis,))

    def augment_samples(self, inputs):
        params = inputs[self.TRANSFORMS]
        if params is None:
            return inputs[self.SAMPLES]
        return keras.ops.where(params["mask"], params["fill"], inputs[self.SAMPLES])

    def get_config(self):
        return {**super().get_config(), **_config(self)}


@helia_export(path="helia_edge.layers.preprocessing.RandomCutout2D")
class RandomCutout2D(BaseAugmentation2D):
    """Replace random rectangular regions in each example during training.

    Regions share their positions across channels. Targets and masks remain
    unchanged; only signal values are occluded. Overlapping regions are allowed.

    Args:
        factor (float | tuple[float, float]): Fractional size bounds in [0, 1] for each spatial axis. A scalar
            sets the upper bound with a zero lower bound; a pair sets both bounds.
        cutouts (int): Nonnegative number of regions sampled per example.
        fill_mode (str): "constant" for a fixed value or "normal" for Gaussian noise.
        fill_value (float): Constant fill value, or the nonnegative standard deviation
            of zero-mean Gaussian noise when fill_mode is "normal".
        **kwargs (Any): Base augmentation options, including seed and data_format.

    Raises:
        ValueError: Size bounds, region count, fill mode or noise deviation
            violate these constraints.
    """

    def __init__(self, factor=0.1, cutouts=1, fill_mode="constant", fill_value=0.0, **kwargs):
        super().__init__(**kwargs)
        _initialize(self, factor, cutouts, fill_mode, fill_value)

    def get_random_transformations(self, input_shape):
        return _parameters(self, input_shape, (self.height_axis, self.width_axis))

    def augment_samples(self, inputs):
        params = inputs[self.TRANSFORMS]
        if params is None:
            return inputs[self.SAMPLES]
        return keras.ops.where(params["mask"], params["fill"], inputs[self.SAMPLES])

    def get_config(self):
        return {**super().get_config(), **_config(self)}
