"""FastEnhancer config validation and preset resolution without optional backends."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from helia_edge.models.fastenhancer_params import (
    FASTENHANCER_PRESET_SOURCE,
    FASTENHANCER_PRESETS,
    FastEnhancerParams,
    resolve_fastenhancer,
)


def test_t_preset_matches_upstream_config():
    t = FASTENHANCER_PRESETS["fastenhancer_t"]
    assert (t.n_fft, t.channels, t.kernel_size, t.stride) == (512, 24, (8, 3, 3), 4)
    rf = t.rnnformer
    assert (rf.num_blocks, rf.channels, rf.freq, rf.num_heads) == (2, 20, 16, 4)
    assert rf.positional_embedding and not rf.attn_bias and not rf.post_act
    assert (t.activation, t.mask, t.input_compression, t.resnet) == ("silu", "none", 0.3, False)
    assert (t.spectral_bins, t.encoder_bins) == (257, 64)


def test_resolve_records_preset_overrides_and_identity():
    official = resolve_fastenhancer("fastenhancer_t")
    assert official.official and official.overrides == {} and official.source == FASTENHANCER_PRESET_SOURCE
    custom = resolve_fastenhancer("fastenhancer_t", {"channels": 32, "rnnformer": {"num_heads": 5}})
    assert not custom.official
    assert custom.params.channels == 32 and custom.params.rnnformer.num_heads == 5
    assert custom.params.rnnformer.channels == 20
    assert type(custom).model_validate_json(custom.model_dump_json()) == custom
    with pytest.raises(ValueError, match="unknown FastEnhancer override"):
        resolve_fastenhancer("fastenhancer_t", {"width": 2})
    with pytest.raises(ValueError, match="unknown FastEnhancer preset"):
        resolve_fastenhancer("fastenhancer_x")


@pytest.mark.parametrize(
    "config",
    [
        {"kernel_size": (6, 3, 3)},
        {"kernel_size": (8, 4, 3)},
        {"n_fft": 510},
        {"n_fft": 520, "stride": 8, "kernel_size": (8, 3)},
        {"rnnformer": {"channels": 20, "num_heads": 3}},
        {"input_compression": 0.0},
        {"form": "trainable"},
        {"width": 2},
        {"channels": "24"},
    ],
)
def test_invalid_configs_fail(config):
    with pytest.raises(ValidationError):
        FastEnhancerParams.from_config(config)


def test_json_roundtrip():
    params = FASTENHANCER_PRESETS["fastenhancer_t"]
    assert FastEnhancerParams.from_config(params.get_config()) == params


def test_params_import_without_backends():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'keras', 'tensorflow', 'torch', 'jax'}:
            raise AssertionError('config imported optional backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
from helia_edge.models import FastEnhancerParams, resolve_fastenhancer
record = resolve_fastenhancer('fastenhancer_t')
assert FastEnhancerParams.model_validate_json(record.params.model_dump_json()) == record.params
assert not any(name in sys.modules for name in ('keras', 'tensorflow', 'torch', 'jax'))
'''
    subprocess.run([sys.executable, "-c", code], check=True, cwd=root,
                   env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True, text=True)
