"""Recipe validation must not require optional execution backends."""

import os
from pathlib import Path
import subprocess
import sys


def test_public_params_validate_without_backend_imports():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'keras', 'tensorflow', 'torch', 'jax'}:
            raise AssertionError('config imported optional backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
from helia_edge.models import MiniResNetV1Params
params = MiniResNetV1Params.from_config({'stacks': 1, 'base_filters': 64})
assert MiniResNetV1Params.model_validate_json(params.model_dump_json()) == params
assert not any(name in sys.modules for name in ('keras', 'tensorflow', 'torch', 'jax'))
'''
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root,
                   env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True)
