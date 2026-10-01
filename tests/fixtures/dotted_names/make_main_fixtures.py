"""Write two fixtures with helia-edge main cb530275 on TensorFlow into tests/fixtures/dotted_names/:

- two_outputs.keras: outputs named 'out.a' and 'out.b', compiled with dict-keyed loss, metrics and
  loss weights; two_outputs_io.npz holds inputs, targets and the TF outputs.
- tcn_layer_norm.keras: a seeded TCN with norm="layer", saved with keras.layers.LayerNormalization over
  spatial axes; tcn_layer_norm_io.npz holds inputs and the TF outputs.

usage: KERAS_BACKEND=tensorflow PYTHONPATH=<main checkout> python make_main_fixtures.py <out dir>
"""

import sys
from pathlib import Path

import keras
import numpy as np

from helia_edge.models import TcnModel, TcnParams

out = Path(sys.argv[1])
rng = np.random.default_rng(11)

keras.utils.set_random_seed(20261001)
inputs = keras.Input((8,), batch_size=None, name="x.in")
hidden = keras.layers.Dense(6, activation="relu", name="hidden.dense")(inputs)
model = keras.Model(
    inputs, {"out.a": keras.layers.Dense(2, name="out.a")(hidden), "out.b": keras.layers.Dense(1, name="out.b")(hidden)}
)
model.compile(
    optimizer="sgd",
    loss={"out.a": "mse", "out.b": "mae"},
    metrics={"out.a": ["mae"], "out.b": ["mse"]},
    loss_weights={"out.a": 1.0, "out.b": 0.5},
)
x = rng.normal(size=(4, 8)).astype(np.float32)
targets = {"out.a": rng.normal(size=(4, 2)).astype(np.float32), "out.b": rng.normal(size=(4, 1)).astype(np.float32)}
y = model.predict(x, verbose=0)
model.save(out / "two_outputs.keras")
np.savez(
    out / "two_outputs_io.npz",
    inputs=x,
    target_a=targets["out.a"],
    target_b=targets["out.b"],
    output_a=y["out.a"],
    output_b=y["out.b"],
)

keras.utils.set_random_seed(20261001)
block = {
    "depth": 1,
    "branch": 1,
    "filters": 8,
    "kernel": (1, 3),
    "dilation": (1, 1),
    "dropout": 0,
    "ex_ratio": 1,
    "se_ratio": 0,
    "norm": "layer",
}
params = TcnParams(
    input_kernel=(1, 3),
    input_norm="layer",
    blocks=[block, {**block, "dilation": (1, 2)}],
    output_kernel=(1, 3),
    include_top=True,
    use_logits=True,
)
tcn = TcnModel.model_from_params(inputs=keras.Input((1, 32, 4), batch_size=1), params=params, num_classes=3)
assert any(type(layer) is keras.layers.LayerNormalization and list(layer.axis) != [-1] for layer in tcn.layers)
xt = rng.normal(size=(2, 1, 1, 32, 4)).astype(np.float32)
yt = np.stack([keras.ops.convert_to_numpy(tcn(xi, training=False)) for xi in xt])
tcn.save(out / "tcn_layer_norm.keras")
np.savez(out / "tcn_layer_norm_io.npz", inputs=xt, outputs=yt)
print("written")
