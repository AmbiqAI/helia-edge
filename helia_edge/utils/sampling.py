"""Generator scheduling policies, independent of training frameworks."""

from enum import StrEnum


class StreamMode(StrEnum):
    GLOBAL = "global"
    FINITE = "finite"
