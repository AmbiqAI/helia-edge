"""Typed parameters of the unext family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UNextBlockParams(BaseModel):
    """UNext block parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        ddepth (int | None): Layer decoder depth
        kernel (int | tuple[int, int]): Kernel size
        pool (int | tuple[int, int]): Pool size
        strides (int | tuple[int, int]): Stride size
        skip (bool): Add skip connection
        expand_ratio (float): Expansion ratio
        se_ratio (float): Squeeze and excite ratio
        dropout (float | None): Dropout rate
        norm (Literal["batch", "layer"] | None): Normalization type

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ddepth: int | None = Field(default=None, description="Layer decoder depth")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    pool: int | tuple[int, int] = Field(default=2, description="Pool size")
    strides: int | tuple[int, int] = Field(default=2, description="Stride size")
    skip: bool = Field(default=True, description="Add skip connection")
    expand_ratio: float = Field(default=1, description="Expansion ratio")
    se_ratio: float = Field(default=0, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")


class UNextParams(BaseModel):
    """UNext parameters

    Attributes:
        family (Literal["unext"]): Model family
        blocks (list[UNextBlockParams]): UNext blocks
        include_top (bool): Include top
        use_logits (bool): Use logits
        output_kernel_size (int | tuple[int, int]): Output kernel size
        output_kernel_stride (int | tuple[int, int]): Output kernel stride
        num_classes (int | None): Classes of the output layer; required with include_top

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["unext"] = "unext"
    blocks: list[UNextBlockParams] = Field(default_factory=list, description="UNext blocks")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(
        default=None, gt=0, description="Classes of the output layer; required with include_top"
    )
    use_logits: bool = Field(default=True, description="Use logits")
    output_kernel_size: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    output_kernel_stride: int | tuple[int, int] = Field(default=1, description="Output kernel stride")
