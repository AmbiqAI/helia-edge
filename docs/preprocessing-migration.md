# Next-major preprocessing migration

This refactor targets the next major release; it does not change the package
version or release existing artifacts. Pin the older EDGE version to replay old
pipelines exactly. Three previously independent portable classes now join the
existing `BaseAugmentation → BaseAugmentation1D/2D` hierarchy. There is no separate
`PortablePreprocessing1D`, private Keras dispatcher, or TensorFlow base class.

## Input and hook contract

`Sample[T].tensor_tree()` returns `TensorPayload[T]`: two-level `signals`,
`targets`, and `masks` dictionaries containing tensors. It creates containers,
never copies tensor storage, and keeps record metadata outside the Keras call.
Tensor inputs and legacy `data` dictionaries remain accepted. Legacy `labels`
and other tensor extras pass through; a lone legacy `targets`/`masks` tensor is
addressed as `default` when selecting aligned keys.

`BaseAugmentationParams` is a frozen Pydantic construction-time schema for seed,
layout, device, mapping mode and aligned key selections. Use
`RandomCrop1D(32, **params.model_dump())`; tensors never enter this validator.
Concrete transforms keep their typed constructor arguments; Keras `get_config`
contains plain values rather than a second object-construction framework.

Signals are converted to compute dtype. Shape contracts are rank2/3 for 1D and
rank3/4 for 2D, with channels-first or channels-last. Targets/masks keep their
dtype. Eager known batch/spatial mismatches are rejected for selected aligned
leaves. Symbolic unknown axes must satisfy the same caller contract at execution;
this is not a dynamic shape-assertion framework. Crop needs static spatial sizes.
RandomChannel retains the image-rank contract of its existing base.

Keep these hooks in custom subclasses:

- `get_random_transformations(batched_shape)` returns parameters or `None`.
- `augment_samples({"data": batch, "transforms": parameters})` handles a batch.
- Alternatively, `augment_sample(...)` handles one sample. Parameter leaves must
  have a batch-leading dimension for mapping. Batchwise parameters must first be
  broadcast along that dimension. The default sequential public `keras.ops.map`
  avoids assumptions about vectorizable callbacks; `auto_vectorize=True` opts in.
- `augment_targets` / `augment_masks` apply selected geometry; defaults reuse the
  same signal transform. Resize uses the explicit target policy below; masks always use nearest index
  selection, retaining even large integer label values without a float round trip.
- `batch_augment(inputs, transformations=None)` is the whole-tree override used
  by composite transforms. Call layer entry points for child Keras tracking.

`call(..., training=None, transformations=None)` belongs to the shared base.
`training_only=True` gates augmentation *before* random sampling. `None`/False
means value identity (signals still follow compute dtype), without advancing RNG;
True augments. Deterministic transforms set `training_only=False` and always run.
There is no mutable `self.training` or `self.backend`. Replace private
`_format_inputs/_format_outputs`, old label hooks and TF-specific `tf_keras_map`
usage with the tensor-tree and target/mask hooks. Base classes are extension
points, not registered concrete saved-model types.

A spatial transform declares `joint=True`. It samples once from the first signal
and shares the parameters across all signal leaves and selected targets/masks.
`aligned_targets=None` / `aligned_masks=None` select all keys; tuples select a
subset, and `()` leaves the group untouched. For example:

```python
from helia_edge.layers.preprocessing import BaseAugmentationParams, RandomCrop1D
params = BaseAugmentationParams(seed=17, aligned_targets=("segmentation",))
crop = RandomCrop1D(32, unique_batch=True, **params.model_dump())
output = crop(sample.tensor_tree(), training=True)
```

`unique_batch=True` means independent offsets per example; False broadcasts one
crop offset to the batch. Call with `transformations=` to reuse already sampled
parameters, bypassing RNG; the caller must supply valid parameter shapes/values.
`AugmentationPipeline(...)(tree, training=flag, transformations=[params, ...])`
accepts one entry per child; `None` lets that child sample normally. Explicit
parameters bypass sampling during training and are ignored at inference by
training-only children, without RNG draws. `force_training=True` deliberately
overrides inference gating.

Crop inference keeps the original shape. Treat random crop as training-data
preprocessing before a fixed-input model, with a separately chosen deterministic
inference window; it is not a shape-stable in-model replacement for that window.
A Functional crop followed by global pooling works with both training modes;
a flatten/dense consumer built for the original length fails when training crops
it. TensorFlow tensor-valued training supports both lengths, but does not remove
that downstream constraint. No implicit center crop is chosen for consumers. Random branches in a pipeline must have
compatible structures/dtypes/output shapes; do not mix arbitrary crops and
identity branches in a compiled conditional expecting a fixed shape.

### Resize target roles

Resizing1D/2D require `target_interpolation="nearest"` or `"signal"` whenever
selected aligned targets are present. The default `None` rejects such targets
rather than guessing their role. `nearest` preserves target dtype and exact label
IDs. `signal` uses the signal interpolation (1D bicubic, 2D configured) and casts
targets to the layer compute dtype, including integer regression inputs; this
conversion may lose large-integer precision intentionally. Masks always use
nearest and retain dtype, independent of target policy. The policy serializes.
For clean ECG regression targets, choose `signal`; categorical segment labels use
`nearest`. One policy applies to selected targets in a layer; separate mixed-role
transforms explicitly using `aligned_targets`, rather than inferring roles from
dtype. Existing resize configs with targets must add the policy when migrating.

## Existing transform inventory and behavior changes

| Group | Migration and explicit changes |
| --- | --- |
| Normalization1D/2D, FirFilter | Shared base, deterministic in inference, retained formulas. FIR coefficient serialization, per-channel taps/layout and auxiliary dtype repairs retained. FIR is same-padded cross-correlation, not SciPy lfilter/filtfilt. |
| LayerNormalization1D/2D, Rescaling1D/2D | Deterministic shared hooks; epsilon now serializes. |
| Resizing1D/2D | Deterministic joint geometry; channels-first singleton axis corrected; 2D interpolation argument now honored and serialized. Selected targets require explicit nearest/signal policy; masks use half-pixel nearest with no numeric casts. |
| RandomGaussianNoise1D | Shared training gate; inference consumes no RNG; signal-only. |
| RandomCrop1D/2D | Batched offset application shared with selected targets/masks; actual training shapes replace misleading old shape overrides. Inference is identity. |
| RandomFlip2D | Training gate now applies; horizontal means width, vertical means height (old axes were reversed). Parameters shared with aligned leaves. |
| RandomChannel | Missing returns repaired; per-example and batchwise channel choice execute; batchwise serializes. Signal-only. |
| RandomCutout1D/2D | Exactly `cutouts` regions, including zero, rather than the former extra initial region. Half-open masks and per-example parameters; signal-only, clean targets/masks unchanged. |
| RandomBackgroundNoises1D | Noise source selection now uses the owned RNG. Noise bank is a nontrainable weight and serializes. |
| RandomSineWave | Multi-channel sine broadcasting repaired; sample rate must be positive. Signal-only augmentation. |
| AddSineWave | Fixed deterministic addition in both training/inference; replaces the old undefined inference result. |
| AmplitudeWarp / RandomNoiseDistortion1D | Public ops, corrected channels-first singleton layout and config typos; configured noise interpolation is honored. Frequency-dependent tensor allocation remains an eager-path qualification, not a fullgraph promise. Supply valid sample-rate/frequency pairs; legacy invalid default combinations are not a supported configuration. |
| FrequencyMixStyle2D | Constructor probability/alpha/epsilon now retain provided values rather than the lower parse bound; config added. Channels-first reduction axes corrected; original mixing formula retained. Signal-only. |
| SpecAugment2D | Shared owned RNG/training/layout contract. Half-open masks, zero-width means no mask, inclusive configured maximum; separate frequency-height/time-width axes and per-example samples. Config preserves superclass values. |
| CascadedBiquadFilter | Causal recurrence unchanged. Optional precomputed SOS serializes and permits loading without SciPy; cutoff-based coefficient design needs SciPy at construction only. Forward/backward is two zero-state passes, not SciPy sosfiltfilt padding. |
| AugmentationPipeline | Serializes concrete child layers; forwards training without Python boolean coercion of tensor flags. Deterministic children still run at inference. |
| RandomChoice / RandomAugmentation1DPipeline/2DPipeline | Captured branch closures, returned outputs, child build/config and owned RNG corrected. Supported batchwise choice is now the default; the previously unimplemented per-example choice explicitly errors. Repeat count zero and rate zero are identity. |

These RNG, gate, axis and mask corrections change frozen augmentation sequences.
They are intentional major-version changes, not bitwise compatibility claims.
No tolerances are relaxed to hide numerical changes. Geometry preserves selected
leaf values by indexing/flipping; signal-only intensity/noise transforms preserve
auxiliary leaves. Layout comparisons and independent references are in the tests.

## Loading and optional pipelines

Use `helia_edge.models.load_model` for concrete registered classes, or import the
classes before direct Keras loading. Normalization2D's old module import remains
an alias. Old simple concrete configs retain their constructor fields, including
PR39's FIR/dtype fixes. Configs requiring retired TFDataLayer/base serialization,
private hooks, unsupported per-example choice, or an exact historic RNG sequence
must run under their original version or be rebuilt explicitly with this guide.
Do not overwrite retained model artifacts during migration. Saving restores layer
config/weights; exact augmentation RNG-position resume is not qualified.

Select KERAS_BACKEND before import. Torch execution/import does not require
TensorFlow. With the TensorFlow backend, map the same layer in tf.data; training
augmentation must use `lambda tree: layer(tree, training=True)`. Cross-global-backend
TF graphs are unsupported. `examples/preprocessing/grain_pipeline.py` uses the
same core with Grain-owned per-record seeds, CPU placement, explicit NumPy output
copies and iterator close; no parallel NumPy transform engine exists.

Qualified locally: Keras3.15.1, TF2.21/Python3.12, Torch2.14/Python3.14, NumPy2.3.3,
CPU, tested layouts/structures, optional Grain0.2.18 workers0/2. Stateful Torch RNG
may require graph breaks; deterministic compile probes use Dynamo/eager. JAX,
GPU, arbitrary nesting, RNG equality across backends, fullgraph stochastic
pipelines and export formats are not certified by these checks.
