"""Typed parameters of the conformer family; importable without Keras."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SubsampleBlockParams(BaseModel):
    """Subsample block parameters

    Attributes:
        depth (int): Depth
        kernel_size (int): Kernel size
        strides (int): Stride size
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int = 256
    kernel_size: int = 3
    strides: int = 2


class ConformerBlockParams(BaseModel):
    """Conformer block parameters

    Attributes:
        depth (int): Depth
        fc_ex_factor (float): FC expansion factor
        fc_res_factor (float): FC residual factor
        embedding (str): Embedding type
        num_heads (int): Number of heads
        kernel_size (int): Kernel size
        dropout (float): Dropout rate
        use_bias (bool): Use bias
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    depth: int = 256
    fc_ex_factor: float = 4
    fc_res_factor: float = 0.5
    embedding: str = "relative"
    num_heads: int = 4
    kernel_size: int = 9
    dropout: float = 0.1
    use_bias: bool = True


class ConformerParams(BaseModel):
    """Conformer parameters

    Attributes:
        family (Literal["conformer"]): Model family
        subsamples (list[SubsampleBlockParams]): Subsample blocks
        blocks (list[ConformerBlockParams]): Conformer blocks
        output_activation (str | None): Output activation
        include_top (bool): Include top
        num_classes (int | None): Classes of the output layer

    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: Literal["conformer"] = "conformer"
    subsamples: list[SubsampleBlockParams] = Field(
        default_factory=list,
        min_length=1,
        validate_default=True,
        description="Subsample blocks",
    )
    blocks: list[ConformerBlockParams] = Field(default_factory=list, description="Conformer blocks")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    num_classes: int | None = Field(default=None, gt=0, description="Classes of the output layer")
