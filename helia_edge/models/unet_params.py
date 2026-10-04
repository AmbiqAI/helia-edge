"""Typed parameters of the unet family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UNetBlockParams(BaseModel):
    """UNet block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        ddepth (int | None): Decoder depth
        kernel (int | tuple[int, int]): Kernel size
        pool (int | tuple[int, int]): Pool size
        strides (int | tuple[int, int]): Stride size
        skip (bool): Add skip connection
        seperable (bool): Use seperable convs
        dropout (float | None): Dropout rate
        norm (Literal["batch", "layer"] | None): Normalization type
        activation (Literal["relu", "relu6", "leaky_relu", "elu", "selu"]): Activation
        dilation (int | tuple[int, int] | None): Dilation factor
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Decoder depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=3, description="Pool size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    seperable: bool = Field(default=False, description="Use seperable convs")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="batch", description="Normalization type")
    activation: Literal["relu", "relu6", "leaky_relu", "elu", "selu"] = Field(default="relu6", description="Activation")
    dilation: int | tuple[int, int] | None = Field(default=None, description="Dilation factor")


class UNetParams(BaseModel):
    """UNet parameters

    Attributes:
        family (Literal["unet"]): Model family
        blocks (list[UNetBlockParams]): UNet blocks
        include_top (bool): Include top
        use_logits (bool): Use logits
        output_kernel_size (int | tuple[int, int]): Output kernel size
        output_kernel_stride (int | tuple[int, int]): Output kernel stride
        num_classes (int | None): Classes of the output layer; required with include_top

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["unet"] = "unet"
    blocks: list[UNetBlockParams] = Field(default_factory=list, description="UNet blocks")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(
        default=None, gt=0, description="Classes of the output layer; required with include_top"
    )
    use_logits: bool = Field(default=True, description="Use logits")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    output_kernel_stride: int | tuple[int, int] = Field(default=1, description="Output kernel stride")
