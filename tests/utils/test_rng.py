"""Sampling probabilities and seeded compatibility for ID schedules."""

from itertools import islice
import random

import pytest

from helia_edge.utils import random_id_generator


@pytest.fixture(autouse=True)
def restore_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


def test_zero_weight_ids_are_never_selected():
    random.seed(0)
    samples = random_id_generator(["excluded", "selected"], weights=[0, 1])
    assert list(islice(samples, 100)) == ["selected"] * 100


def test_relative_float_weights_control_sampling():
    random.seed(0)
    samples = list(islice(random_id_generator([0, 1], weights=[0.1, 0.9]), 10000))
    assert 8500 < samples.count(1) < 9500


@pytest.mark.parametrize("mass", [5e-324, 1.0, 1e308])
def test_zero_weight_exclusion_is_independent_of_scale(mass):
    random.seed(0)
    samples = random_id_generator(["selected", "excluded"], weights=[mass, 0])
    assert list(islice(samples, 100)) == ["selected"] * 100


def test_unweighted_seeded_sequence_is_preserved():
    random.seed(0)
    assert list(islice(random_id_generator([10, 20]), 10)) == [20, 20, 10, 20, 20, 20, 20, 20, 20, 10]


@pytest.mark.parametrize(
    "weights", [[1], [1, 2, 3], [-1, 2], [0, 0], [float("nan"), 1], [float("inf"), 1], [1e308, 1e308]]
)
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        next(random_id_generator([0, 1], weights=weights))
