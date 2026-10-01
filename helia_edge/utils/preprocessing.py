"""Preprocessing bounds; the tf.data adapters live in ``helia_edge.data.tf_data``."""

from __future__ import annotations

from ..data.tf_data import convert_inputs_to_tf_dataset as convert_inputs_to_tf_dataset
from ..data.tf_data import create_dataset_from_data as create_dataset_from_data
from ..data.tf_data import create_interleaved_dataset_from_generator as create_interleaved_dataset_from_generator
from ..data.tf_data import get_output_signature as get_output_signature
from ..data.tf_data import get_output_signature_from_fn as get_output_signature_from_fn
from ..data.tf_data import get_output_signature_from_gen as get_output_signature_from_gen


def parse_factor(
    param: float | tuple[float | None, float] | list[float | None],
    min_value: float | None = 0.0,
    max_value: float | None = 1.0,
    param_name: str = "factor",
) -> tuple[float, float]:
    """Normalize scalar or paired bounds and validate their range."""
    low, high = (min_value, param) if isinstance(param, (float, int)) else param
    if high is None:
        raise ValueError(f"{param_name} requires an upper bound")
    low = high if low is None else low
    if low > high:
        raise ValueError(f"{param_name}[0] must be <= {param_name}[1]; got {param}")
    if (min_value is not None and low < min_value) or (max_value is not None and high > max_value):
        raise ValueError(f"{param_name} must be inside [{min_value}, {max_value}]; got {param}")
    return low, high
