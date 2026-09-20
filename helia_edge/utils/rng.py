"""Random seeds and ID schedules."""

import random
from collections.abc import Generator, MutableSequence, Sequence


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
    weights: list[int] | None = None,
) -> Generator[T, None, None]:
    """Sample IDs uniformly with replacement indefinitely."""
    while True:
        yield random.choice(ids)
