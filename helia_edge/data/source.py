"""Random-access record sources."""

from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class DataSource(Protocol[T_co]):
    """Records addressed by index, such as a list or a Grain source.

    ``source[i]`` must return the same record for the same ``i`` so that a seed fixes the order
    and the augmentation of every epoch.
    """

    def __len__(self) -> int:
        """Return the number of records."""
        ...

    def __getitem__(self, index: int) -> T_co:
        """Return record ``index``, for ``0 <= index < len(self)``."""
        ...
