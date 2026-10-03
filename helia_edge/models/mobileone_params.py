"""Typed parameters of the mobileone family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MobileOneBlockParams(BaseModel):
    """MobileOne block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        padding (int | tuple[int, int]): Padding size
        se_ratio (float): Squeeze Excite ratio
        se_depth (int): Depth length to apply SE
        num_conv_branches (int): Number of conv branches
        activation (str): Activation function

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    padding: int | tuple[int, int] = Field(default=0, description="Padding size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    se_depth: int = Field(default=0, description="Depth length to apply SE")
    num_conv_branches: int = Field(default=2, description="# conv branches")
    activation: str = Field(default="relu6", description="Activation function")


class MobileOneParams(BaseModel):
    """MobileOne parameters

    Attributes:
        family (Literal["mobileone"]): Model family
        blocks (list[MobileOneBlockParams]): MobileOne blocks
        input_filters (int): Input filters
        input_kernel_size (int | tuple[int, int]): Input kernel size
        input_strides (int | tuple[int, int]): Input stride
        input_padding (int | tuple[int, int]): Input padding
        include_top (bool): Include top
        output_activation (str | None): Output activation
        dropout (float): Dropout rate
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["mobileone"] = "mobileone"
    blocks: list[MobileOneBlockParams] = Field(default_factory=list, description="MobileOne blocks")

    input_filters: int = Field(default=3, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_padding: int | tuple[int, int] = Field(default=1, description="Input padding")

    # output_filters: int = Field(default=0, description="Output filters")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, description="Classes of the output layer")
    output_activation: str | None = Field(default=None, description="Output activation")
    dropout: float = Field(default=0.2, description="Dropout rate")
    # drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
