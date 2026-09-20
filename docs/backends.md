# Backend installation and support

Install the capability you use:

```sh
pip install 'helia-edge[tensorflow]'
# or, in a separate environment:
pip install 'helia-edge[torch]'
```

Set `KERAS_BACKEND=tensorflow` or `KERAS_BACKEND=torch` before importing Keras or
accessing a Keras feature. Base `helia-edge` installs no training framework;
file/factory/logger helpers work independently. Existing public attribute paths
remain available and load their dependencies when accessed.

This changes installation behavior: existing TensorFlow users must select the
`tensorflow` extra. `litert` includes TensorFlow conversion plus the LiteRT runtime.
`metal` is opt-in and must be combined with `tensorflow`; Metal compatibility has
not been validated for this baseline.

The tested baseline is Linux CPU, Python 3.12.5, Keras 3.15.1, TensorFlow 2.21.0 or
Torch 2.14.0+cpu, and LiteRT 2.2.0 for conversion. CI passes on Python 3.12–3.13
for TensorFlow and 3.12–3.14 for Torch in separate environments. TensorFlow 2.21
has no Python 3.14 wheel; the `tensorflow` and `litert` extras require Python below
3.14 for this baseline. Other platforms, GPU builds
and newer dependency resolutions need their own validation. To reproduce the CPU
Torch environment, install `torch==2.14.0` from
`https://download.pytorch.org/whl/cpu` before installing the Torch extra.

Portable metrics, TCN construction, EMA quantization and masked-autoencoder
training have isolated Torch checks. TFDataLayer-based preprocessing,
generator-to-tf.data utilities, contrastive training, TFLite/LiteRT conversion and FLOP profiling remain
TensorFlow-specific at this milestone. Backend availability does not certify every
model, trainer, precision, compiled/distributed configuration or export format.

## Loading saved custom objects

Lazy imports no longer register all custom Keras classes as a side effect of
`import helia_edge`. The EDGE model loader registers supported objects explicitly:

```python
from helia_edge.models import load_model
model = load_model('model.keras')
```

For direct Keras loading, import the custom classes used by the model, or register
supported EDGE classes before loading:

```python
import helia_edge
helia_edge.register_keras_serializables()
import keras
model = keras.saving.load_model('model.keras')
```

Registration initializes the selected backend. It does not enable unsafe loading
or make TensorFlow-only custom layers usable on Torch. Fresh-process round trips
cover an EDGE EMA quantizer on both backends and legacy normalization on TF.

## Validation

CI includes a base-only lane and separate backend environments. Import checks
assert the opposite framework is absent. TF runs the full suite, including float
and int8 conversion; Torch runs the portable subset and custom-object reloads.
CPU tests set `CUDA_VISIBLE_DEVICES=-1` so installed GPU drivers cannot affect the
CPU export path. Existing augmentation layers still use private Keras internals;
their TF regressions are covered, but their portability is separate future work.

## Masked-autoencoder training

`MaskedAutoencoder` and its patch layers use public `keras.ops` and `keras.random`
for the forward path, with separate TensorFlow and Torch gradient application.
Use normal Keras `compile(loss='mse', optimizer='adam', jit_compile=False)` and
`fit(x)`; targets are generated from masked input patches. Explicit `y` and
`sample_weight` are rejected rather than silently ignored. Native loops can apply
their own objective and weighting:

```python
targets, predictions = model.reconstruction_targets(x, training=True)
# Apply your native TF/Torch objective and optimizer to these tensors.
```

`call()` returns the same `Reconstruction(targets, predictions)` named tuple.
`calculate_loss(x, test=False)` returns `ReconstructionLoss(loss, targets, predictions)`
using the compiled objective. Both retain tuple unpacking. Keras layers receive the training flag; plain
single-argument callables retain their original calling convention and must own
their state behavior. Use serializable Keras layers to save complete models.

`MaskedPatchEncoder2D(seed=...)` tracks a Keras seed generator. Inference disables
Dropout/BatchNorm training behavior but still samples masks. Same-seed random
streams are tested within each backend; cross-backend numerical checks use fixed
masks, inputs and weights. Whole RNG/sampler resume and identical stochastic
training trajectories are not promised. Save/reload tests cover full-model
weights, controlled-mask predictions and optimizer slots/iteration on each backend.

Build the model before constructing a native Torch optimizer so every parameter
is present. The regression fixtures in `tests/trainers/` exercise native TF/Torch
optimization. Run their forward/gradient/update comparison with:

```sh
python tests/trainers/compare_mae_backends.py \
  --tensorflow-python /path/to/tf-env/bin/python \
  --torch-python /path/to/torch-env/bin/python
```

CI compares evidence produced in the separate backend jobs at `rtol=1e-5`,
`atol=1e-6`. This is a small float32 CPU fixture, not certification of mixed
precision, Torch compilation, distribution or MAE export. SleepKit currently has
no MAE consumer; this component is not counted as a two-KIT shared-API proof.
