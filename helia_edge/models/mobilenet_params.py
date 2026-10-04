"""Typed parameters of the mobilenet family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MobileNetV1Params(BaseModel):
    """MobileNetV1 parameters

    Attributes:
        family (Literal["mobilenet"]): Model family
        input_filters (int): Input filters
        input_strides (int | tuple[int, int]): Input stride
        include_top (bool): Include top
        output_activation (str | None): Output activation
        num_classes (int | None): Classes of the output layer; required with include_top

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["mobilenet"] = "mobilenet"
    input_filters: int = Field(default=8, description="Input filters")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(
        default=None, gt=0, description="Classes of the output layer; required with include_top"
    )
    output_activation: str | None = Field(default=None, description="Output activation")
