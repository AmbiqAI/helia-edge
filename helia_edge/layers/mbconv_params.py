"""Typed MBConv block parameters; importable without Keras."""

from pydantic import BaseModel, ConfigDict, Field


class MBConvParams(BaseModel):
    """MBConv parameters

    Attributes:
        filters (int): Number of filters
        depth (int): Layer depth
        ex_ratio (float): Expansion ratio
        kernel_size (int | tuple[int, int]): Kernel size
        strides (int | tuple[int, int]): Stride size
        se_ratio (float): Squeeze Excite ratio
        droprate (float): Drop rate
        bn_momentum (float): Batch normalization momentum
        activation (str): Activation function
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    ex_ratio: float = Field(default=1, description="Expansion ratio")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")
    bn_momentum: float = Field(default=0.9, description="Batch normalization momentum")
    activation: str = Field(default="relu6", description="Activation function")
