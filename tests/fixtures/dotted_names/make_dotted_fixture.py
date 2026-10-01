"""Write tests/fixtures/dotted_names/: a seeded EfficientNetV2 .keras saved by helia-edge main a677c85b on
TensorFlow (its layer names contain '.', as models from earlier helia-edge versions do) and the TF outputs.

usage: KERAS_BACKEND=tensorflow PYTHONPATH=<main checkout> python make_dotted_fixture.py <out dir>
"""

import sys
from pathlib import Path

import keras
import numpy as np

from helia_edge.models import EfficientNetParams, EfficientNetV2Model

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
keras.utils.set_random_seed(20261001)
params = EfficientNetParams(
    input_filters=8,
    input_kernel_size=(1, 3),
    input_strides=(1, 2),
    blocks=[
        {"filters": 8, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "se_ratio": 2},
        {"filters": 16, "depth": 1, "kernel_size": (1, 3), "strides": (1, 2), "ex_ratio": 2, "se_ratio": 2},
    ],
    output_filters=16,
)
model = EfficientNetV2Model.model_from_params(
    inputs=keras.Input((1, 32, 4), batch_size=1), params=params, num_classes=3
)
dotted = sorted(layer.name for layer in model.layers if "." in layer.name)
assert dotted, "the fixture must contain dotted layer names"
x = np.random.default_rng(7).normal(size=(2, 1, 1, 32, 4)).astype(np.float32)
y = np.stack([keras.ops.convert_to_numpy(model(xi, training=False)) for xi in x])
model.save(out / "efficientnetv2.keras")
np.savez(out / "efficientnetv2_io.npz", inputs=x, outputs=y)
print(f"{len(dotted)} dotted layer names, e.g. {dotted[:3]}")
