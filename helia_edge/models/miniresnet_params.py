"""Lightweight validated MiniResNet-v1 architecture config; no backend imports."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MiniResNetV1Params(BaseModel):
    """Validated architecture config, independent of inputs and weight assets.

    Each stack has two residual blocks; stack widths double from base_filters.
    Changing defaults defines a new architecture, not a pretrained variant.
    Flatten pooling requires fixed spatial dimensions. Backbone trainability
    defaults to True; freeze layers explicitly when fine-tuning a new head.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stacks: int = Field(default=1, ge=1, le=3)
    base_filters: int = Field(default=64, gt=0)
    pooling: Literal["flatten", "avg", "max"] = "flatten"
    dropout: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    output_activation: Literal["softmax", "sigmoid", "linear"] = "softmax"
    name: str = Field(default="miniresnet_v1", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-compatible constructor config without weights."""
        return self.model_dump(mode="json")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MiniResNetV1Params":
        """Validate external config, rejecting unknown keys and coercions."""
        return cls.model_validate(dict(config))

