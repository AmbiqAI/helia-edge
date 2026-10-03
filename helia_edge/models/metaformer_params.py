"""Typed parameters of the metaformer family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NameArgs(BaseModel):
    """Name and arguments

    Attributes:
        name (str): Name
        args (dict): Arguments

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="conv", description="Name")
    args: dict = Field(default_factory=dict, description="Arguments")


class MetaFormerBlockParams(BaseModel):
    """MetaFormer block parameters

    Attributes:
        layers (int): Number of layers
        patch_embed (dict): Patch embedding
        token_mixer (NameArgs): Token mixer
        channel_mixer (NameArgs): Channel mixer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layers: int = Field(default=2, description="Number of layers")
    patch_embed: dict = Field(default_factory=dict, description="Patch embedding")
    token_mixer: NameArgs = Field(default_factory=dict, description="Token mixer")
    channel_mixer: NameArgs = Field(default_factory=dict, description="Channel mixer")


class MetaFormerParams(BaseModel):
    """MetaFormer parameters

    Attributes:
        family (Literal["metaformer"]): Model family
        blocks (list[MetaFormerBlockParams]): MetaFormer blocks
        output_filters (int): Output filters
        output_activation (str | None): Output activation
        include_top (bool): Include top
        dropout (float): Dropout rate
        drop_connect_rate (float): Drop connect rate
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["metaformer"] = "metaformer"
    blocks: list[MetaFormerBlockParams] = Field(default_factory=list, description="MetaFormer blocks")
    output_filters: int = Field(default=0, description="Output filters")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
