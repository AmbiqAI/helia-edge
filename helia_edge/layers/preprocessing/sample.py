"""Typed containers for tensor payloads; record metadata stays with the caller."""

from dataclasses import dataclass, field
from typing import Generic, TypedDict, TypeVar

T = TypeVar("T")


class TensorPayload(TypedDict, Generic[T]):
    """Two-level tensor-only schema accepted by portable preprocessing layers."""

    signals: dict[str, T]
    targets: dict[str, T]
    masks: dict[str, T]


@dataclass(frozen=True)
class Sample(Generic[T]):
    """Live tensor facade, not a serializable model config or an immutable tensor.

    Conversion creates new dictionaries but never copies tensor storage. Keys in
    each group name tensor leaves. Keep record IDs and other metadata outside this
    object, and pass ``sample.tensor_tree()`` to a Keras layer.
    """

    signals: dict[str, T]
    targets: dict[str, T] = field(default_factory=dict)
    masks: dict[str, T] = field(default_factory=dict)

    def tensor_tree(self) -> TensorPayload[T]:
        """Return fresh containers referencing the original tensor leaves."""
        return {"signals": dict(self.signals), "targets": dict(self.targets), "masks": dict(self.masks)}
