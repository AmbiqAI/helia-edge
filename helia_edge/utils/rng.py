"""Random seeds and ID schedules."""

import random
from collections.abc import Generator, MutableSequence, Sequence
from itertools import accumulate
from math import isfinite


def set_random_seed(seed: int | None = None) -> int:
    """Set Python, NumPy and selected Keras backend seeds; return the seed."""

    import keras

    seed = random.randint(0, 2**16) if seed is None else seed
    random.seed(seed)
    keras.utils.set_random_seed(seed)
    return seed


def uniform_id_generator[T](
    ids: MutableSequence[T],
    repeat: bool = True,
    shuffle: bool = True,
) -> Generator[T, None, None]:
    """Yield each ID once per cycle; shuffle mutates the supplied sequence."""
    while True:
        if shuffle:
            random.shuffle(ids)
        yield from ids
        if not repeat:
            break


def random_id_generator[T](
    ids: Sequence[T],
    weights: Sequence[float] | None = None,
) -> Generator[T, None, None]:
    """Sample with replacement, using optional nonnegative relative weights."""
    cumulative = None
    if weights is not None:
        if len(weights) != len(ids):
            raise ValueError("weights must have one value per ID")
        if any(not isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("weights must be finite and nonnegative")
        cumulative = tuple(accumulate(weights))
        if not cumulative or cumulative[-1] <= 0 or not isfinite(cumulative[-1]):
            raise ValueError("weights must have a finite positive total")
        # Unit total prevents subnormal rounding from selecting zero-weight IDs.
        cumulative = tuple(value / cumulative[-1] for value in cumulative)
    while True:
        yield random.choice(ids) if cumulative is None else random.choices(ids, cum_weights=cumulative, k=1)[0]
