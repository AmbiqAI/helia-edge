"""Shared public-Keras boundary for shape-preserving 1D signal transforms."""

import keras


class PortablePreprocessing1D(keras.Layer):
    """Transform signals while leaving labels, targets and masks unchanged.

    Accepts a tensor, a legacy ``{"data": tensor, ...}`` dictionary, or a
    ``TensorPayload``. Signal leaves are (T, C)/(B, T, C), or channels-first
    equivalents. Tensor-only auxiliary groups are passed through unchanged.
    ``seed``, ``auto_vectorize`` and ``device`` retain constructor compatibility;
    these batched transforms do not need per-example vectorization.
    """

    SAMPLES = "data"
    TRANSFORMS = "transforms"

    def __init__(self, seed=None, auto_vectorize=True, data_format=None, device="cpu", **kwargs):
        # Cast signal leaves explicitly, never every tensor in a structured input.
        kwargs.setdefault("autocast", False)
        super().__init__(**kwargs)
        self.seed = seed
        self.generator = keras.random.SeedGenerator(seed)
        self.auto_vectorize = auto_vectorize
        self.data_format = data_format or keras.backend.image_data_format()
        if self.data_format not in ("channels_first", "channels_last"):
            raise ValueError("data_format must be channels_first or channels_last")
        self.device = device

    @property
    def random_generator(self):
        return self.generator

    @property
    def data_axis(self):
        return -1 if self.data_format == "channels_first" else -2

    def _transform(self, samples, training):
        return self.augment_samples({self.SAMPLES: samples})

    def _signal(self, samples, training):
        samples = keras.ops.convert_to_tensor(samples)
        rank = len(samples.shape)
        if rank not in (2, 3):
            raise ValueError(f"Expected rank 2 or 3 signal, received shape {samples.shape}")
        samples = keras.ops.cast(samples, self.compute_dtype)
        if rank == 2:
            samples = keras.ops.expand_dims(samples, 0)
        outputs = self._transform(samples, training)
        return keras.ops.squeeze(outputs, 0) if rank == 2 else outputs

    def call(self, inputs, training=True):
        with keras.device(self.device):
            if not isinstance(inputs, dict):
                return self._signal(inputs, training)
            if "signals" in inputs:
                if "data" in inputs:
                    raise ValueError("Use either signals or legacy data, not both")
                signals = inputs["signals"]
                if not isinstance(signals, dict) or not signals:
                    raise ValueError("signals must be a nonempty dictionary of tensor leaves")
                for group in ("targets", "masks"):
                    if group in inputs and not isinstance(inputs[group], dict):
                        raise ValueError(f"{group} must be a dictionary of tensor leaves")
                return {**inputs, "signals": {key: self._signal(value, training) for key, value in signals.items()}}
            if "data" not in inputs:
                raise ValueError("Expected a signals or legacy data key")
            return {**inputs, "data": self._signal(inputs["data"], training)}

    def get_config(self):
        return {
            **super().get_config(),
            "seed": self.seed,
            "auto_vectorize": self.auto_vectorize,
            "data_format": self.data_format,
            "device": self.device,
        }
