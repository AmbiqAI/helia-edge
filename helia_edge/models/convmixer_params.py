"""Typed parameters of the ConvMixer family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConvMixerParams(BaseModel):
    """ConvMixer parameters

    Attributes:
        family (Literal["convmixer"]): Model family
        filters (int): Number of filters per layer
        depth (int): Network depth
        kernel_size (int): Filter size
        patch_size (int): Patch size
        include_top (bool): Include top
        num_classes (int | None): Classes of the dense head; None for no dense layer
        output_activation (str | None): Output activation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["convmixer"] = "convmixer"
    filters: int = Field(default=256, description="# filters per layer")
    depth: int = Field(default=8, description="Network depth")
    kernel_size: int = Field(default=5, description="Filter size")
    patch_size: int = Field(default=2, description="Patch size")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, description="Classes of the dense head; None for no dense layer")
    output_activation: str | None = Field(default=None, description="Output activation")
