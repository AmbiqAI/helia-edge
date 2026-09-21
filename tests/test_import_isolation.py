"""Run imports in fresh processes so module caches cannot hide eager dependencies."""

import importlib.util
import os
import subprocess
import sys

import pytest


def run_python(source):
    result = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_base_helpers_do_not_import_training_frameworks():
    run_python("""
import sys
import helia_edge as helia
from helia_edge.utils.file import compute_checksum
assert callable(helia.utils.setup_logger)
assert callable(helia.utils.env_flag)
assert callable(helia.utils.ItemFactory)
assert callable(helia.utils.create_factory)
assert list(helia.utils.uniform_id_generator([1, 2], repeat=False, shuffle=False)) == [1, 2]
assert helia.utils.parse_factor((None, 0.5)) == (0.5, 0.5)
assert not {'keras', 'tensorflow', 'torch'} & sys.modules.keys()
""")


def test_missing_backend_guidance():
    if importlib.util.find_spec("keras") is not None:
        pytest.skip("This check belongs in the base-only environment")
    run_python("""
import sys
import helia_edge as helia
try:
    helia.models.TcnModel
except ImportError as exc:
    assert 'helia-edge[' in str(exc), str(exc)
    if sys.version_info >= (3, 14):
        assert 'Python 3.12–3.13' in str(exc), str(exc)
        assert 'KERAS_BACKEND=torch' in str(exc), str(exc)
else:
    raise AssertionError('Missing backend unexpectedly available')
""")


def test_selected_backend_does_not_require_the_other():
    if importlib.util.find_spec("keras") is None:
        pytest.skip("Training extra is not installed")
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    other = "tensorflow" if backend == "torch" else "torch"
    if os.environ.get("HELIA_REQUIRE_ISOLATION") == "1":
        assert importlib.util.find_spec(other) is None
    run_python(f"""
import sys
import helia_edge as helia
assert callable(helia.models.TcnModel.model_from_params)
assert callable(helia.models.load_model)
assert callable(helia.metrics.MultiF1Score)
from helia_edge.losses.simclr import SimCLRLoss
assert callable(SimCLRLoss)
assert callable(helia.layers.EmaResidualVectorQuantizer)
helia.register_keras_serializables()
assert {other!r} not in sys.modules
""")


def test_seed_zero_is_not_replaced():
    if importlib.util.find_spec("keras") is None:
        pytest.skip("Training extra is not installed")
    run_python("""
from helia_edge.utils import set_random_seed
assert set_random_seed(0) == 0
""")
