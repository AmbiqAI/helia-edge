# MLPerf Tiny reference architectures

These four constructors reproduce the **scale-1 architecture** of the pinned
benchmark captures. They initialize new weights. They do not download official
trained weights, reproduce reported accuracy, perform task preprocessing, or
constitute a compliant MLPerf submission. Scaling is intentionally omitted in this
first version; there are no alternate-width defaults or generic architecture
parameters.

```python
import keras
from helia_edge.models import (
    mlperf_tiny_kws, mlperf_tiny_vww, mlperf_tiny_resnet, mlperf_tiny_ad,
)

keras.utils.set_random_seed(20260926)
model = mlperf_tiny_kws()
```

| Constructor | Input excluding batch | Output | Exact architecture |
| --- | --- | --- | --- |
| `mlperf_tiny_kws` | 49×10×1 features | 12 probabilities | DS-CNN64, 10×4 stride2 stem, four 3×3 depthwise/1×1 pointwise pairs; 25×5 pool |
| `mlperf_tiny_vww` | 96×96×3 RGB | 2 probabilities | MobileNetV1 alpha0.25: width8 stem, 13 depthwise/pointwise pairs, final width256; 3×3 pool |
| `mlperf_tiny_resnet` | 32×32×3 RGB | 10 probabilities | Reduced three-stack ResNet28/56/112; nine convolutions, three residual additions; 8×8 pool |
| `mlperf_tiny_ad` | 640 features | 640 linear reconstructions | Dense128×4 → 8 → 128×4 → 640; BN/ReLU on every hidden layer |

Spatial constructors explicitly use channels-last, independent of the global Keras
image layout setting. All convolutions use the reference's SAME padding and biases. Activations are
ordinary ReLU, not ReLU6. Batch normalization retains momentum0.99/epsilon0.001.
KWS retains dropout0.2 and0.4; inference disables it. VWW has no dropout. ResNet
projection shortcuts have no normalization, and the second main-path convolution
has normalization but no activation until after addition. AD's bottleneck also
has BN/ReLU; the last Dense is linear and no anomaly score is computed.

The reference depthwise kernels use GlorotUniform with no depthwise regularizer.
Older training scripts pass `kernel_initializer`/`kernel_regularizer` to
DepthwiseConv2D; their serialized saved configurations reveal that these did not
set its depthwise initializer/regularizer. The constructors use explicit Keras3
arguments matching those saved configurations, rather than translating the
misleading keyword names into a different model.

## Pinned sources and captured identity

Training source/configuration reference:
[`mlcommons/tiny` at `4addd0fa08d216e20637637874e084895f289da4`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training).
Capture corpus:
[`AmbiqAI/helia-model-zoo` at `a5c3073ee55a9afd413430b1191adfea658ae009`](https://github.com/AmbiqAI/helia-model-zoo/tree/a5c3073ee55a9afd413430b1191adfea658ae009).
`tests/fixtures/mlperf-tiny-reference.json` retains serialized layer configuration,
connectivity and captured operator shapes/options, without trained weight arrays.
Source/artifact hashes are in `examples/mlperf_tiny/references.json`.

| Model | Pinned source/configuration | Captured INT8 SHA256 |
| --- | --- | --- |
| KWS | `keyword_spotting/keras_model.py`; `trained_models/kws_ref_model/saved_model.pb` configuration | `aeea436800704fce17b17292e4412630ad856e9d777c044c64ef748a880bd0ae` |
| VWW | `visual_wake_words/vww_model.py`; `trained_models/vww_96.h5` configuration | `597a384c8c2c8a1276f04702f25013b7838f2f814f1ca7c174d295b73e3d6b7b` |
| ResNet | `image_classification/keras_model.py:resnet_v1_eembc(conv_filters=28)` and captured graph | `a65375297130b602a5b28523c20d9e24127cc36508c5f67586f02c01eb6299e6` |
| AD | `anomaly_detection/keras_model.py`; `trained_models/ad01.h5` configuration | `87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24` |

Three upstream INT8 artifacts (KWS/VWW/AD) match these corpus hashes exactly.
The captured ResNet uses28/56/112 channels. The pinned upstream source supports
that parameterization, but its presently stored pretrained artifacts use16 or40
base channels and have different hashes. The reference fixture therefore adapts
the serialized16-channel topology to the source function's `conv_filters=28`
parameter and independently checks it against the exact captured graph. This
establishes topology, not the unknown training run or original weights lineage
of the28-channel capture. Do not substitute the upstream smaller model or call
this ResNet18.

KWS has another concrete source/configuration difference: the pinned training
script computes a24×5 pool from floor(input/2), whereas the pinned saved model and
captured graph use25×5. The constructor follows the saved artifact's25×5 pool.
These distinctions are explicit so a source refresh cannot silently change the
benchmark architecture.

## Local fixture generation and checks

With the repository's `litert` extra, Python3.12/3.13 and TensorFlow backend,
produce one retained FP32 fixture from an installed checkout:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. python examples/mlperf_tiny/generate.py \
  --model kws --output /absolute/external/path/kws-fixture
```

Use `vww`, `resnet` or `ad` for the other constructors. The output directory must
be new. It retains the initialized Keras model, exact model config, FP32 export,
zero/signal input arrays, complete LiteRT outputs and Keras reference outputs,
source/seed/dependency identities, licenses and hashes. LiteRT BUILTIN_REF runs
single-thread without delegates; exported FP32 outputs must match Keras with
`rtol=1e-5, atol=1e-5`. Seed alone is not a cross-version byte identity guarantee.
To make initialized KWS/VWW models numerically discriminating, the helper increases
the synthetic signal amplitude by powers of ten until its output differs from the
zero case by at least0.01. It records the amplitude and fails if no finite signal
qualifies. These diagnostic inputs are not representative task data; weights and
architecture remain unchanged. Tests reject a converter returning the same output
for both cases.
The helper uses the TensorFlow backend; the constructors use ordinary Keras APIs.
No calibration/training campaign or extra precision matrix is implicit.

The tests compare full layer semantics/connectivity to independent serialized
reference configurations, check seeded reconstruction and serialization, and
compare real exported operator topology/shapes/options to the captured graphs.
The converter may fold Flatten into FullyConnected and omit optional zero Dense
biases. The comparison normalizes only singleton-spatial flattening and verifies
that every omitted source bias is zero; source-layer checks still require the
original Flatten and bias settings. Weights, quantization parameters and opcode versions vary with precision/runtime
and are not treated as architecture identity. Plausible wrong pooling, ReLU6,
missing residuals, output activation and ResNet width are deliberately rejected.

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. pytest -q \
  tests/models/test_mlperf_tiny.py tests/converters/test_mlperf_tiny_export.py
```

## Licenses

MLCommons Tiny source is Apache-2.0. The anomaly detector includes Hitachi's2020
MIT copyright/license. Original attributions, license texts and modification
notice are shipped in `helia_edge/models/licenses/`, including in the wheel.
Trained assets remain external; no official model weights are bundled.
