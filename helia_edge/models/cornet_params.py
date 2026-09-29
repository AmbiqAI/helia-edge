"""Validated CorNET architecture config; no backend imports."""

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CORNET_SOURCE = "Biswas et al., IEEE TBioCAS 13(2):282-291, 2019, doi:10.1109/TBCAS.2019.2892297"


class CorNetParams(BaseModel):
    """CorNET heart-rate regressor: two convolution stages then stacked LSTMs.

    Defaults are the paper's HR network (Sec. III, Fig. 6, Table III): two
    Conv1D(32, 40) stages with batch normalization, ReLU, max pooling 4 and
    dropout 0.1, then LSTM(128) twice and a single linear output. The paper
    publishes no code or weights; changed values are new architectures.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    conv_stages: int = Field(default=2, ge=1)
    filters: int = Field(default=32, gt=0)
    kernel_size: int = Field(default=40, gt=0)
    pool_size: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1, allow_inf_nan=False)
    lstm_layers: int = Field(default=2, ge=1)
    lstm_units: int = Field(default=128, gt=0)
    recurrent_activation: Literal["sigmoid", "hard_sigmoid"] = "sigmoid"
    name: str = Field(default="cornet", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-compatible constructor config without weights."""
        return self.model_dump(mode="json")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CorNetParams":
        """Validate external config, rejecting unknown keys and coercions."""
        return cls.model_validate_json(json.dumps(dict(config)))
