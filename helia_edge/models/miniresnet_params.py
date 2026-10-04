"""Lightweight validated MiniResNet-v1 architecture config; no backend imports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MiniResNetV1Params(BaseModel):
    """Validated architecture config, independent of inputs and weight assets.

    Each stack has two residual blocks; stack widths double from base_filters.
    Changing defaults defines a new architecture, not a pretrained variant.
    Flatten pooling requires fixed spatial dimensions. Backbone trainability
    defaults to True; freeze layers explicitly when fine-tuning a new head.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["miniresnet"] = "miniresnet"
    stacks: int = Field(default=1, ge=1, le=3)
    base_filters: int = Field(default=64, gt=0)
    pooling: Literal["flatten", "avg", "max"] = "flatten"
    dropout: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    output_activation: Literal["softmax", "sigmoid", "linear"] = "softmax"
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer; required")
