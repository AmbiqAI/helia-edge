"""Typed parameters of the composer family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ComposerLayerParams(BaseModel):
    """Composer layer parameters

    Attributes:
        name (str): Layer name
        params (dict): Layer arguments
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Layer name")
    params: dict = Field(default_factory=dict, description="Layer arguments")


class ComposerParams(BaseModel):
    """Composer Network parameters

    Attributes:
        family (Literal["composer"]): Model family
        layers (list[ComposerLayerParams]): Network layers
        include_top (bool): Include top
        output_activation (str | None): Output activation
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["composer"] = "composer"
    layers: list[ComposerLayerParams] = Field(default_factory=list, description="Network layers")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer")
    output_activation: str | None = Field(default=None, description="Output activation")
