"""Typed parameters of the TCN family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _positive_spatial(value):
    if value is not None:
        dimensions = (value,) if isinstance(value, int) else value
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("Spatial dimensions must be positive")
    return value


class TcnBlockParams(BaseModel):
    """TCN block parameters

    Attributes:
        depth (int): Layer depth
        branch (int): Number of branches
        filters (int): Number of filters
        kernel (int | tuple[int, int]): Kernel size
        dilation (int | tuple[int, int]): Dilation rate
        ex_ratio (float): Expansion ratio
        se_ratio (float): Squeeze and excite ratio
        dropout (float | None): Dropout rate
        norm (Literal["batch", "layer"] | None): Normalization type
        activation (str): Activation function
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int = Field(default=1, gt=0, description="Layer depth")
    branch: int = Field(default=1, gt=0, description="Number of branches")
    filters: int = Field(..., gt=0, description="# filters")
    kernel: int | tuple[int, int] = Field(default=3, description="Kernel size")
    dilation: int | tuple[int, int] = Field(default=1, description="Dilation rate")
    ex_ratio: float = Field(default=1, gt=0, allow_inf_nan=False, description="Expansion ratio")
    se_ratio: float = Field(default=0, ge=0, allow_inf_nan=False, description="Squeeze and excite ratio")
    dropout: float | None = Field(default=None, ge=0, lt=1, allow_inf_nan=False, description="Dropout rate")
    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    activation: str = Field(default="relu6", description="Activation function")

    _spatial = field_validator("kernel", "dilation")(_positive_spatial)


class TcnParams(BaseModel):
    """TCN parameters

    Attributes:
        family (Literal["tcn"]): Model family
        input_kernel (int | tuple[int, int] | None): Input kernel size
        input_norm (Literal["batch", "layer"] | None): Input normalization type
        block_type (Literal["lg", "mb", "sm"]): Block type
        blocks (list[TcnBlockParams]): TCN blocks
        output_kernel (int | tuple[int, int]): Output kernel size
        include_top (bool): Include top
        num_classes (int | None): Classes of the output layer; required with include_top
        use_logits (bool): Use logits
        output_activation (str | None): Output activation
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["tcn"] = "tcn"
    input_kernel: int | tuple[int, int] | None = Field(default=None, description="Input kernel size")
    input_norm: Literal["batch", "layer"] | None = Field(default="layer", description="Input normalization type")
    block_type: Literal["lg", "mb", "sm"] = Field(default="mb", description="Block type")
    blocks: list[TcnBlockParams] = Field(default_factory=list, description="TCN blocks")
    output_kernel: int | tuple[int, int] = Field(default=3, description="Output kernel size")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(
        default=None, gt=0, description="Classes of the output layer; required with include_top"
    )
    use_logits: bool = Field(default=True, description="Use logits")
    output_activation: str | None = Field(default=None, description="Output activation")

    _spatial = field_validator("input_kernel", "output_kernel")(_positive_spatial)


def compact_tcn_params(*, filters: int = 8, num_classes: int | None = None) -> TcnParams:
    """Four small SE4 blocks with 1/2/4/8 dilations and per-point linear output.

    The input shape is supplied to ``build`` by the caller. At least eight channels retain the builder's
    SE squeeze path.
    """
    if type(filters) is not int or filters < 8:
        raise ValueError("compact TCN filters must be an integer >= 8")
    return TcnParams(
        input_kernel=None,
        input_norm="batch",
        block_type="sm",
        blocks=[
            TcnBlockParams(
                filters=filters,
                kernel=(1, 3),
                dilation=(1, dilation),
                depth=1,
                branch=1,
                ex_ratio=1,
                se_ratio=4,
                dropout=None,
                norm="batch",
                activation="relu6",
            )
            for dilation in (1, 2, 4, 8)
        ],
        output_kernel=(1, 1),
        include_top=True,
        num_classes=num_classes,
        use_logits=True,
        output_activation=None,
    )
