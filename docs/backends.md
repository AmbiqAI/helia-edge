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
Torch 2.14.0+cpu, and LiteRT 2.2.0 for conversion. CI also exercises Python 3.13 and
3.14 in separate TF-only and Torch-only environments. Other platforms, GPU builds
and newer dependency resolutions need their own validation. To reproduce the CPU
Torch environment, install `torch==2.14.0` from
`https://download.pytorch.org/whl/cpu` before installing the Torch extra.

Portable metrics, TCN construction and EMA quantization have isolated Torch checks.
TFDataLayer-based preprocessing, generator-to-tf.data utilities, contrastive and
masked-autoencoder trainers, TFLite/LiteRT conversion and FLOP profiling remain
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
