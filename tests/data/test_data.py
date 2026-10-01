"""helia_edge.data without frameworks: imports, argument checks and the tf.data re-exports."""

import subprocess
import sys

import pytest

from helia_edge.data import to_grain


def run_python(source: str) -> None:
    result = subprocess.run([sys.executable, "-I", "-c", source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_package_imports_no_framework():
    run_python("""
import sys
import helia_edge as helia
from helia_edge.data import DataSource, to_grain, to_tf_dataset, to_torch_loader
from helia_edge.utils import preprocessing
assert callable(helia.data.create_interleaved_dataset_from_generator)
assert not {'keras', 'tensorflow', 'torch', 'grain'} & sys.modules.keys()
""")


def test_missing_grain_names_the_extra():
    run_python("""
import importlib.abc
import sys
class NoGrain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'grain':
            raise ModuleNotFoundError('absent', name='grain')
sys.meta_path.insert(0, NoGrain())
from helia_edge.data import to_grain
try:
    to_grain([1, 2])
except ImportError as exc:
    assert 'helia-edge[grain]' in str(exc), str(exc)
else:
    raise AssertionError('to_grain ran without Grain')
""")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"shuffle": True}, "needs a seed"),
        ({"transform": lambda record, rng: record}, "needs a seed"),
        ({"batch_size": 0}, "batch_size"),
        ({"num_epochs": 0}, "num_epochs"),
        ({"workers": -1}, "workers"),
        ({"workers": True}, "workers"),
        ({"worker_buffer_size": 0}, "worker_buffer_size"),
        ({"batch_size": 2.5}, "batch_size"),
        ({"num_epochs": 2.0}, "num_epochs"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"seed": 1.5, "shuffle": True}, "seed"),
    ],
)
def test_invalid_arguments_are_refused_before_grain_is_needed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        to_grain([1, 2, 3], **kwargs)


def test_utils_preprocessing_reexports_the_tf_data_helpers():
    from helia_edge.data import tf_data
    from helia_edge.utils import preprocessing

    names = [
        "convert_inputs_to_tf_dataset",
        "create_dataset_from_data",
        "create_interleaved_dataset_from_generator",
        "get_output_signature",
        "get_output_signature_from_fn",
        "get_output_signature_from_gen",
    ]
    for name in names:
        assert getattr(preprocessing, name) is getattr(tf_data, name)
        assert getattr(__import__("helia_edge").utils, name) is getattr(tf_data, name)
