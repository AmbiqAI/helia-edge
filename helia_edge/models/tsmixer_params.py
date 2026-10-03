"""Typed parameters of the tsmixer family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TsMixerBlockParams(BaseModel):
    """TsMixer block parameters

    Attributes:
        norm (Literal["batch", "layer"]): Normalization type
        activation (Literal["relu", "gelu"]): Activation type
        dropout (float): Dropout rate
        ff_dim (int): Feed forward dimension
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    activation: Literal["relu", "gelu"] | None = Field(default="relu", description="Activation type")
    dropout: float | None = Field(default=None, description="Dropout rate")
    ff_dim: int | None = Field(default=None, description="Feed forward dimension")


class TsMixerParams(BaseModel):
    """TsMixer parameters

    Attributes:
        family (Literal["tsmixer"]): Model family
        blocks (list[TsBlockParams]): TsMixer blocks
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["tsmixer"] = "tsmixer"
    blocks: list[TsMixerBlockParams] = Field(default_factory=list, description="UNext blocks")
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer")
