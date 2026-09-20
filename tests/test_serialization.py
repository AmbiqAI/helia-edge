"""Exercise EDGE custom objects across fresh-process safe reloads."""

import os
import subprocess
import sys

import keras
import numpy as np
import pytest

from helia_edge.layers import EmaResidualVectorQuantizer


@pytest.mark.parametrize("layer_name", ["quantizer", "normalization"])
def test_custom_layer_safe_reload(tmp_path, layer_name):
    if layer_name == "normalization":
        if keras.backend.backend() != "tensorflow":
            pytest.skip("Legacy TFDataLayer augmentations remain TF-only")
        from helia_edge.layers.preprocessing import Normalization1D

        layer = Normalization1D(mean=1.0, variance=4.0)
    else:
        layer = EmaResidualVectorQuantizer(num_levels=1, num_embeddings=4, embedding_dim=2)
    model = keras.Sequential([keras.Input(shape=(3, 2)), layer])
    x = np.arange(12, dtype="float32").reshape(2, 3, 2)
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    path = tmp_path / "custom.keras"
    model.save(path)
    np.save(tmp_path / "input.npy", x)
    np.save(tmp_path / "expected.npy", expected)
    source = """
import sys
from pathlib import Path
import keras
import numpy as np
from helia_edge.models import load_model
path = Path(sys.argv[1])
model = load_model(path / 'custom.keras')
actual = keras.ops.convert_to_numpy(model(np.load(path / 'input.npy'), training=False))
np.testing.assert_allclose(actual, np.load(path / 'expected.npy'), atol=1e-6)
"""
    result = subprocess.run(
        [sys.executable, "-c", source, str(tmp_path)], capture_output=True, text=True, timeout=60, env=os.environ.copy()
    )
    assert result.returncode == 0, result.stdout + result.stderr
