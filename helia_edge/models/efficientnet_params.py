"""Typed parameters of the efficientnet family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..layers.mbconv_params import MBConvParams


class EfficientNetParams(BaseModel):
    """EfficientNet parameters

    Attributes:
        family (Literal["efficientnet"]): Model family
        blocks (list[MBConvParams]): EfficientNet blocks
        input_filters (int): Input filters
        input_kernel_size (int | tuple[int, int]): Input kernel size
        input_strides (int | tuple[int, int]): Input stride
        input_activation (str): Input activation
        output_filters (int): Output filters
        output_activation (str | None): Output activation
        include_top (bool): Include top
        dropout (float): Dropout rate
        drop_connect_rate (float): Drop connect rate
        use_logits (bool): Use logits
        activation (str): Activation function
        norm (Literal["batch", "layer"] | None): Normalization type
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["efficientnet"] = "efficientnet"
    blocks: list[MBConvParams] = Field(default_factory=list, description="EfficientNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_kernel_size: int | tuple[int, int] = Field(default=3, description="Input kernel size")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    input_activation: str = Field(default="relu6", description="Input activation")
    output_filters: int = Field(default=0, description="Output filters")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, description="Classes of the output layer")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    use_logits: bool = Field(default=True, description="Use logits")
    activation: str = Field(default="relu6", description="Activation function")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
