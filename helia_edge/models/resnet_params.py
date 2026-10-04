"""Typed parameters of the resnet family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResNetBlockParams(BaseModel):
    """ResNet block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        bottleneck (bool): Use bottleneck blocks
        activation (str): Activation function

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    bottleneck: bool = Field(default=False, description="Use bottleneck blocks")
    activation: str = Field(default="relu6", description="Activation function")


class ResNetParams(BaseModel):
    """ResNet parameters

    Attributes:
        family (Literal["resnet"]): Model family
        blocks (list[ResNetBlockParams]): ResNet blocks
        input_filters (int): Input filters
        input_kernel_size (int | tuple[int, int]): Input kernel size
        input_strides (int | tuple[int, int]): Input stride
        input_activation (str): Input activation
        include_top (bool): Include top
        output_activation (str | None): Output activation
        dropout (float): Dropout rate
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["resnet"] = "resnet"
    blocks: list[ResNetBlockParams] = Field(default_factory=list, description="ResNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_activation: str = Field(default="relu6", description="Input activation")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer")
    output_activation: str | None = Field(default=None, description="Output activation")
    dropout: float = Field(default=0.2, description="Dropout rate")
