"""Training backends supported by EDGE's portable components."""

from enum import StrEnum


class Backend(StrEnum):
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
