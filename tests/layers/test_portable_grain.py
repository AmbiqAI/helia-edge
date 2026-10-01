"""Optional integration: real Grain workers call the portable EDGE layers."""

import importlib.util
import os
import sys
from pathlib import Path

import keras
import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "preprocessing"


def test_grain_worker_rng_and_auxiliary_leaves(monkeypatch):
    pytest.importorskip("grain")
    if keras.backend.backend() != "torch":
        pytest.skip("Qualified CPU Torch/Grain path")
    # Grain workers import the example's transform by module name, so they need its directory too.
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(EXAMPLE_DIR), os.environ.get("PYTHONPATH")])))
    from grain_pipeline import owned_records

    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    records = [
        {"signals": {"ecg": x + n}, "targets": {"seg": x.astype(np.int32)}, "masks": {"valid": x % 2 == 0}}
        for n in range(4)
    ]
    a = owned_records(records, seed=42, workers=0)
    b = owned_records(records, seed=42, workers=2)
    c = owned_records(records, seed=42, workers=2)
    changed = owned_records(records, seed=43, workers=0)
    for index, (aa, bb, cc) in enumerate(zip(a, b, c, strict=True)):
        for av, bv, cv in zip(keras.tree.flatten(aa), keras.tree.flatten(bb), keras.tree.flatten(cc), strict=True):
            np.testing.assert_array_equal(av, bv)
            np.testing.assert_array_equal(bv, cv)
        np.testing.assert_array_equal(bb["targets"]["seg"], records[index]["targets"]["seg"])
        np.testing.assert_array_equal(bb["masks"]["valid"], records[index]["masks"]["valid"])
        assert bb["targets"]["seg"].dtype == np.int32
        assert bb["masks"]["valid"].dtype == np.bool_
    assert any(not np.array_equal(x["signals"]["ecg"], y["signals"]["ecg"]) for x, y in zip(a, changed, strict=True))
    assert "tensorflow" not in sys.modules and importlib.util.find_spec("tensorflow") is None
