"""Unified public-Keras hooks for deterministic preprocessing and augmentation."""

from typing import Literal

import keras
from pydantic import BaseModel, ConfigDict


class BaseAugmentationParams(BaseModel):
    """Shared construction-time configuration; never validates live tensors."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int | None = None
    auto_vectorize: bool = False
    data_format: Literal["channels_first", "channels_last"] = "channels_last"
    device: str = "cpu"
    aligned_targets: tuple[str, ...] | None = None
    aligned_masks: tuple[str, ...] | None = None


class BaseAugmentation(keras.Layer):
    """Base for rank-specific transforms, with one sampling/application contract.

    Override ``augment_samples`` or ``augment_sample`` and optionally
    ``get_random_transformations``. Sample parameters once per signal batch;
    joint transforms reuse them across aligned signals, targets and masks.
    Per-example parameter leaves must have a leading batch axis. Deterministic
    subclasses set ``training_only=False``. Bases are not serialized transforms.
    """

    SAMPLES = "data"
    LABELS = "labels"
    TARGETS = "targets"
    TRANSFORMS = "transforms"
    NDIMS = 4
    training_only = True
    joint = False

    def __init__(
        self,
        seed=None,
        auto_vectorize=False,
        data_format=None,
        device="cpu",
        aligned_targets=None,
        aligned_masks=None,
        **kwargs,
    ):
        kwargs.setdefault("autocast", False)
        super().__init__(**kwargs)
        params = BaseAugmentationParams(
            seed=seed,
            auto_vectorize=auto_vectorize,
            data_format=data_format or keras.backend.image_data_format(),
            device=device,
            aligned_targets=aligned_targets,
            aligned_masks=aligned_masks,
        )
        self.seed = params.seed
        self.auto_vectorize = params.auto_vectorize
        self.data_format = params.data_format
        self.device = params.device
        self.aligned_targets = params.aligned_targets
        self.aligned_masks = params.aligned_masks
        self.generator = keras.random.SeedGenerator(self.seed)

    @property
    def random_generator(self):
        return self.generator

    @property
    def ch_axis(self):
        return -(self.NDIMS - 1) if self.data_format == "channels_first" else -1

    @property
    def data_axis(self):
        return -1 if self.data_format == "channels_first" else -2

    @property
    def height_axis(self):
        return -2 if self.data_format == "channels_first" else -3

    @property
    def width_axis(self):
        return -1 if self.data_format == "channels_first" else -2

    def _map_fn(self, func, inputs):
        if self.auto_vectorize:
            return keras.ops.vectorized_map(func, inputs)
        return keras.ops.map(func, inputs)

    def get_random_transformations(self, input_shape):
        """Return batched parameter tensors, or None for deterministic transforms."""
        return None

    def augment_sample(self, inputs):
        raise NotImplementedError("Implement augment_sample or augment_samples")

    def augment_samples(self, inputs):
        if inputs[self.TRANSFORMS] is None:
            return self._map_fn(
                lambda x: self.augment_sample({self.SAMPLES: x, self.TRANSFORMS: None}), inputs[self.SAMPLES]
            )
        return self._map_fn(self.augment_sample, inputs)

    def augment_targets(self, inputs):
        """Apply geometry to selected targets using the signal parameters."""
        return self.augment_samples(inputs)

    def augment_masks(self, inputs):
        """Apply geometry to selected masks using the signal parameters."""
        return self.augment_targets(inputs)

    def _batch(self, value, *, signal):
        value = keras.ops.convert_to_tensor(value)
        rank = len(value.shape)
        if rank not in (self.NDIMS - 1, self.NDIMS):
            raise ValueError(f"Expected rank {self.NDIMS - 1} or {self.NDIMS}, received {value.shape}")
        if signal:
            value = keras.ops.cast(value, self.compute_dtype)
        unbatched = rank == self.NDIMS - 1
        return (keras.ops.expand_dims(value, 0) if unbatched else value), unbatched

    def _aligned(self, value, reference):
        for axis, (size, expected) in enumerate(zip(value.shape, reference.shape, strict=True)):
            if axis == self.ch_axis % self.NDIMS:
                continue
            if size is not None and expected is not None and size != expected:
                raise ValueError("Aligned leaves must match batch and spatial dimensions")

    def _apply(self, value, params, method, *, signal, reference=None):
        batch, unbatched = self._batch(value, signal=signal)
        if reference is not None:
            self._aligned(batch, reference)
        output = method({self.SAMPLES: batch, self.TRANSFORMS: params})
        return keras.ops.squeeze(output, 0) if unbatched else output

    def batch_augment(self, inputs, transformations=None):
        """Apply to a tensor or schema, with optional pre-sampled parameters."""
        is_dict = isinstance(inputs, dict)
        if is_dict and "signals" in inputs and "data" in inputs:
            raise ValueError("Use either signals or legacy data, not both")
        canonical = is_dict and "signals" in inputs
        if canonical:
            signals = inputs["signals"]
            if not isinstance(signals, dict) or not signals:
                raise ValueError("signals must be a nonempty dictionary of tensor leaves")
        elif is_dict:
            if self.SAMPLES not in inputs:
                raise ValueError("Expected signals or legacy data")
            signals = {"default": inputs[self.SAMPLES]}
        else:
            signals = {"default": inputs}
        reference, _ = self._batch(next(iter(signals.values())), signal=True)
        shared = transformations
        if self.joint and shared is None:
            shared = self.get_random_transformations(keras.ops.shape(reference))
        result = {}
        for key, value in signals.items():
            batch, _ = self._batch(value, signal=True)
            params = (
                shared
                if self.joint or transformations is not None
                else self.get_random_transformations(keras.ops.shape(batch))
            )
            result[key] = self._apply(
                value, params, self.augment_samples, signal=True, reference=reference if self.joint else None
            )
        if not is_dict:
            return result["default"]
        output = {**inputs, **({"signals": result} if canonical else {self.SAMPLES: result["default"]})}
        if self.joint:
            for group, keys, method in (
                ("targets", self.aligned_targets, self.augment_targets),
                ("masks", self.aligned_masks, self.augment_masks),
            ):
                if group not in inputs:
                    continue
                leaves = inputs[group]
                mapping = isinstance(leaves, dict)
                if canonical and not mapping:
                    raise ValueError(f"{group} must be a dictionary of tensor leaves")
                leaves = leaves if mapping else {"default": leaves}
                selected = tuple(leaves) if keys is None else keys
                if any(key not in leaves for key in selected):
                    raise ValueError(f"Unknown aligned key in {group}: {selected}")
                converted = dict(leaves)
                for key in selected:
                    converted[key] = self._apply(leaves[key], shared, method, signal=False, reference=reference)
                output[group] = converted if mapping else converted["default"]
        return output

    def _identity(self, inputs, *, validate_rank=True):
        def identity(x):
            return x[self.SAMPLES]

        def convert(value):
            if not validate_rank:
                # Generic composites defer spatial rank validation to their children.
                return keras.ops.cast(keras.ops.convert_to_tensor(value), self.compute_dtype)
            return self._apply(value, None, identity, signal=True)

        if not isinstance(inputs, dict):
            return convert(inputs)
        if "signals" in inputs and "data" in inputs:
            raise ValueError("Use either signals or legacy data, not both")
        if "signals" in inputs:
            if not isinstance(inputs["signals"], dict) or not inputs["signals"]:
                raise ValueError("signals must be a nonempty dictionary of tensor leaves")
            return {**inputs, "signals": {key: convert(value) for key, value in inputs["signals"].items()}}
        if self.SAMPLES not in inputs:
            raise ValueError("Expected signals or legacy data")
        return {**inputs, self.SAMPLES: convert(inputs[self.SAMPLES])}

    def call(self, inputs, training=None, transformations=None):
        with keras.device(self.device):
            if not self.training_only:
                return self.batch_augment(inputs, transformations=transformations)
            if training is False or training is None:
                return self._identity(inputs)
            if training is True:
                return self.batch_augment(inputs, transformations=transformations)
            return keras.ops.cond(
                training,
                lambda: self.batch_augment(inputs, transformations=transformations),
                lambda: self._identity(inputs),
            )

    def get_config(self):
        return {
            **super().get_config(),
            "seed": self.seed,
            "auto_vectorize": self.auto_vectorize,
            "data_format": self.data_format,
            "device": self.device,
            "aligned_targets": self.aligned_targets,
            "aligned_masks": self.aligned_masks,
        }


class BaseAugmentation1D(BaseAugmentation):
    """One-dimensional signals with optional batch axis."""

    NDIMS = 3


class BaseAugmentation2D(BaseAugmentation):
    """Two-dimensional images with optional batch axis."""

    NDIMS = 4
