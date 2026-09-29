"""Generator scheduling policies, independent of training frameworks."""

from enum import StrEnum


class StreamMode(StrEnum):
    """Select how sample identifiers are scheduled across generator partitions.

    Attributes:
        GLOBAL: Use a caller-owned schedule, including repeated or weighted
            identifiers, without partitioning by worker count.
        FINITE: Partition a finite identifier sequence into contiguous shards
            for terminating generators.
    """

    GLOBAL = "global"
    FINITE = "finite"
