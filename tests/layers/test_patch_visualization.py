"""Patch visualization is optional; patch tensor computation is independent."""

import subprocess
import sys

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from helia_edge.layers import PatchLayer2D


def test_visualization_returns_valid_batch_index(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    layer = PatchLayer2D(4, 4, 1, 2, 2)
    images = keras.ops.convert_to_tensor(np.arange(16, dtype="float32").reshape(1, 4, 4, 1))
    patches = layer(images)
    assert layer.show_patched_image(images, patches) == 0
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(layer.reconstruct_from_patch(patches[0])), keras.ops.convert_to_numpy(images[0])
    )
    plt.close("all")


def test_missing_visualization_dependency_preserves_patch_computation():
    source = """
import importlib.abc
import sys
class NoMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'matplotlib':
            raise ModuleNotFoundError('no matplotlib', name='matplotlib')
sys.meta_path.insert(0, NoMatplotlib())
import keras
import numpy as np
from helia_edge.layers import PatchLayer2D
layer = PatchLayer2D(4, 4, 1, 2, 2)
images = keras.ops.convert_to_tensor(np.arange(16, dtype='float32').reshape(1, 4, 4, 1))
patches = layer(images)
np.testing.assert_allclose(keras.ops.convert_to_numpy(layer.reconstruct_from_patch(patches[0])), keras.ops.convert_to_numpy(images[0]))
try:
    layer.show_patched_image(images, patches)
except ImportError as exc:
    assert 'helia-edge[plotting]' in str(exc), str(exc)
else:
    raise AssertionError('Visualization should require the plotting extra')
"""
    result = subprocess.run([sys.executable, "-I", "-c", source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
