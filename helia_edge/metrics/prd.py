"""Percent RMS Difference metrics."""

import keras

from ..utils import helia_export


@helia_export(path="helia_edge.metrics.PRD", package="metrics")
class PRD(keras.Metric):
    """Percent RMS difference metric with optional energy normalization.

    PRD is computed as:
        100 * sqrt(sum((y_true - y_pred)^2) / denom)

    where denom is either:
    - sum(y_true^2) when ``normalized=True``
    - N (number of elements) when ``normalized=False``
    """

    def __init__(self, normalized: bool = True, name: str = "prd", **kwargs):
        super().__init__(name=name, **kwargs)
        self.normalized = normalized
        self._num = self.add_variable(shape=(), initializer="zeros", name="num")
        self._den = self.add_variable(shape=(), initializer="zeros", name="den")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.convert_to_tensor(y_true, dtype=self.dtype)
        y_pred = keras.ops.convert_to_tensor(y_pred, dtype=self.dtype)

        err_sq = keras.ops.square(y_true - y_pred)

        if sample_weight is not None:
            sample_weight = keras.ops.convert_to_tensor(sample_weight, dtype=self.dtype)
            sample_weight = keras.ops.broadcast_to(sample_weight, keras.ops.shape(err_sq))
            err_sq = err_sq * sample_weight

        num = keras.ops.sum(err_sq)

        if self.normalized:
            den_terms = keras.ops.square(y_true)
            if sample_weight is not None:
                den_terms = den_terms * sample_weight
            den = keras.ops.sum(den_terms)
        else:
            if sample_weight is not None:
                den = keras.ops.sum(sample_weight)
            else:
                den = keras.ops.cast(keras.ops.size(y_true), self.dtype)

        self._num.assign_add(num)
        self._den.assign_add(den)

    def result(self):
        eps = keras.ops.cast(keras.backend.epsilon(), self.dtype)
        ratio = self._num / (self._den + eps)
        ratio = keras.ops.maximum(ratio, keras.ops.cast(0.0, self.dtype))
        return keras.ops.cast(100.0, self.dtype) * keras.ops.sqrt(ratio)

    def reset_state(self):
        for v in self.variables:
            v.assign(keras.ops.zeros(v.shape, dtype=v.dtype))

    def get_config(self):
        return {"normalized": self.normalized, **super().get_config()}


@helia_export(path="helia_edge.metrics.TruePRD", package="metrics")
class TruePRD(PRD):
    """Compatibility wrapper for normalized PRD."""

    def __init__(self, name: str = "true_prd", **kwargs):
        kwargs.pop("normalized", None)
        super().__init__(normalized=True, name=name, **kwargs)
