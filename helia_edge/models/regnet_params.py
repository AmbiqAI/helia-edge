"""Typed parameters of the regnet family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RegNetBlockParam(BaseModel):
    """RegNet block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        group_width (int): Group width
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        se_ratio (float): Squeeze Excite ratio
        droprate (float): Drop rate
        activation (str): Activation function

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    group_width: int = Field(default=1, description="Group width. Must be divisible by in/out filters")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")
    activation: str = Field(default="relu6", description="Activation function")


class RegNetParams(BaseModel):
    """RegNet parameters

    Attributes:
        family (Literal["regnet"]): Model family

        blocks (list[RegNetBlockParam]): RegNet blocks
        input_filters (int): Input filters
        input_strides (int | tuple[int, int]): Input stride
        input_activation (str): Input activation
        output_filters (int): Output filters
        block_style (Literal["y", "z"]): Block style
        include_top (bool): Include top
        output_activation (str | None): Output activation
        dropout (float): Dropout rate
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["regnet"] = "regnet"
    blocks: list[RegNetBlockParam] = Field(default_factory=list, description="RegNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_activation: str = Field(default="relu6", description="Input activation")
    output_filters: int = Field(default=0, description="Output filters")
    block_style: Literal["y", "z"] = Field(default="y", description="Block style")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer")
    output_activation: str | None = Field(default=None, description="Output activation")
    dropout: float = Field(default=0.2, description="Dropout rate")
