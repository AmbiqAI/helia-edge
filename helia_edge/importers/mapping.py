"""Declarative weight mappings from a source file to a Keras model; importable without Keras."""

from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..export.recipe import SHA256


class Transpose(BaseModel):
    """Permute the axes, as ``numpy.transpose``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["transpose"] = "transpose"
    perm: tuple[int, ...]

    def apply(self, x: npt.NDArray) -> npt.NDArray:
        return np.transpose(x, self.perm)


class Reshape(BaseModel):
    """Reshape in C order, as ``numpy.reshape``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reshape"] = "reshape"
    shape: tuple[int, ...]

    def apply(self, x: npt.NDArray) -> npt.NDArray:
        return np.reshape(x, self.shape)


class GateReorder(BaseModel):
    """Reorder equal gate blocks along ``axis``, for example ONNX LSTM ``"iofc"`` to Keras ``"ifco"``.

    ``source`` and ``target`` name the same gates, one letter each, in their stored orders.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["gate_reorder"] = "gate_reorder"
    source: str
    target: str
    axis: int = 0

    @model_validator(mode="after")
    def _same_gates(self) -> "GateReorder":
        if sorted(self.source) != sorted(self.target) or len(set(self.source)) != len(self.source):
            raise ValueError(f"Gate orders {self.source!r} and {self.target!r} must name the same distinct gates")
        return self

    def apply(self, x: npt.NDArray) -> npt.NDArray:
        if x.shape[self.axis] % len(self.source):
            raise ValueError(f"Axis {self.axis} of shape {x.shape} does not split into {len(self.source)} gates")
        blocks = dict(zip(self.source, np.split(x, len(self.source), axis=self.axis), strict=True))
        return np.concatenate([blocks[gate] for gate in self.target], axis=self.axis)


class Split(BaseModel):
    """Take part ``index`` of ``parts`` equal parts along ``axis``.

    A source tensor may feed several rows only through splits that together use every part once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["split"] = "split"
    axis: int
    parts: int = Field(ge=2)
    index: int = Field(ge=0)

    @model_validator(mode="after")
    def _index_in_range(self) -> "Split":
        if self.index >= self.parts:
            raise ValueError(f"Split index {self.index} is not below parts {self.parts}")
        return self

    def apply(self, x: npt.NDArray) -> npt.NDArray:
        if x.shape[self.axis] % self.parts:
            raise ValueError(f"Axis {self.axis} of shape {x.shape} does not split into {self.parts} parts")
        return np.split(x, self.parts, axis=self.axis)[self.index]


Transform = Annotated[Transpose | Reshape | GateReorder | Split, Field(discriminator="kind")]


class WeightRow(BaseModel):
    """One model weight: its source tensors, how they combine, and the transforms to its layout.

    Attributes:
        sources: Source tensor names. Several sources are combined by ``combine`` before ``transforms``.
        combine: ``sum`` (for example an LSTM's input and recurrent biases) or ``concat`` along
            ``concat_axis``; required with several sources.
        transforms: Applied in order to the (combined) source. A ``Split`` may only come first and
            only with one source.
        layer: Name of the Keras layer holding the weight (``outer/inner`` for a nested model).
        weight: Name of the weight within the layer, such as ``kernel`` or ``bias``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: tuple[str, ...] = Field(min_length=1)
    combine: Literal["sum", "concat"] | None = None
    concat_axis: int = 0
    transforms: tuple[Transform, ...] = ()
    layer: str
    weight: str

    @model_validator(mode="after")
    def _consistent(self) -> "WeightRow":
        if len(self.sources) > 1 and self.combine is None:
            raise ValueError(f"Sources {list(self.sources)} need combine='sum' or 'concat'")
        if len(self.sources) == 1 and self.combine is not None:
            raise ValueError("combine applies to several sources")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"Sources {list(self.sources)} repeat")
        splits = [i for i, t in enumerate(self.transforms) if isinstance(t, Split)]
        if splits and (splits != [0] or len(self.sources) > 1):
            raise ValueError("A split must be the first transform of a single-source row")
        return self

    @property
    def split(self) -> Split | None:
        first = self.transforms[0] if self.transforms else None
        return first if isinstance(first, Split) else None

    def value(self, tensors: dict[str, npt.NDArray]) -> npt.NDArray:
        """This row's weight value from the source tensors."""
        values = [np.asarray(tensors[name]) for name in self.sources]
        if self.combine == "sum":
            x = np.sum(values, axis=0)
        elif self.combine == "concat":
            x = np.concatenate(values, axis=self.concat_axis)
        else:
            x = values[0]
        for transform in self.transforms:
            x = transform.apply(x)
        return x


class SourcePin(BaseModel):
    """The one source file a mapping was written for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    sha256: SHA256
    note: str = ""


class WeightMapping(BaseModel):
    """Every weight of one architecture, from one source format.

    Attributes:
        name: Mapping name.
        format: Source format, a key of ``helia_edge.registry.importers``.
        source: The pinned source file.
        rows: One row per model weight.
        unused: Source tensors deliberately not imported (every other one must be used).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    format: str
    source: SourcePin
    rows: tuple[WeightRow, ...] = Field(min_length=1)
    unused: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _targets_once(self) -> "WeightMapping":
        targets = [(row.layer, row.weight) for row in self.rows]
        repeated = sorted({t for t in targets if targets.count(t) > 1})
        if repeated:
            raise ValueError(f"Weights {repeated} are mapped more than once")
        return self
