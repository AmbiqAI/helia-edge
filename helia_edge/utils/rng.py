"""
# Random Number Generator API

This module provides utility functions to set random seed and create random id generators.

Functions:
    set_random_seed: Set random seed across libraries: TF, Numpy, Python
    uniform_id_generator: Simple generator that yields ids in a uniform manner
    random_id_generator: Simple generator that yields ids in a random manner

"""

import random
from typing import Generator, Iterable, TypeVar
import keras


T = TypeVar("T")


def set_random_seed(seed: int | None = None) -> int:
    """Set random seed across libraries: TF, Numpy, Python

    Args:
        seed (int | None, optional): Random seed state to use. Defaults to None.

    Returns:
        int: Random seed
    """

    seed = seed or random.randint(0, 2**16)
    random.seed(seed)
    # keras will set all backends including numpy
    keras.utils.set_random_seed(seed)
    return seed


def uniform_id_generator(
    ids: Iterable[T],
    repeat: bool = True,
    shuffle: bool = True,
) -> Generator[T, None, None]:
    """Simple generator that yields ids in a uniform manner.

    Args:
        ids (Iterable[T]): List of ids.
        repeat (bool, optional): Whether to repeat generator. Defaults to True.
        shuffle (bool, optional): Whether to shuffle ids.. Defaults to True.

    Note:
        If repeat is False, generator will stop after yielding all ids once.
        If shuffle, ids parameter will be modified in place.

    Returns:
        Generator[T, None, None]: Generator
    Yields:
        T: Id
    """
    while True:
        if shuffle:
            random.shuffle(ids)
        yield from ids
        if not repeat:
            break
        # END IF
    # END WHILE


def random_id_generator(
    ids: Iterable[T],
    weights: list[int] | None = None,
) -> Generator[T, None, None]:
    """Simple generator that yields ids in a random manner.

    Args:
        ids (Iterable[T]): List of ids
        weights (list[int], optional): Weights for each id. Defaults to None.

    Returns:
        Generator[T, None, None]: Generator

    Yields:
        T: Id
    """
    while True:
        yield random.choice(ids)
    # END WHILE
