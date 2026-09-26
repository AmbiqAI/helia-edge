# Compact TCN performance fixtures

This example produces **synthetic, untrained performance fixtures**, not task-quality
models. It reuses `TcnModel` with four small depthwise/pointwise blocks, batch
normalization and ReLU6, dilations 1/2/4/8, SE ratio 4, and widths 8 and 16.
`recipe.json` records the per-stage kernel/dilation controls and all builder defaults.
The width variants replace every block's `filters` field. Inputs are `[1,240,14]`;
outputs are `[1,240,2]` logits. Same padding and global SE pooling require whole
windows: there is no causal/streaming or recurrent-state claim.

Use Python 3.12 or 3.13 and the TensorFlow Keras backend with the repository's
`litert` extra. From an installed checkout, with its source on `PYTHONPATH`:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. python examples/compact_tcn/generate.py \
  --output /absolute/external/path/tcn-fixtures
KERAS_BACKEND=tensorflow PYTHONPATH=. python examples/compact_tcn/generate.py \
  --output /absolute/external/path/tcn-fixtures --verify
```

The output directory must not exist for generation; partial failures are preserved
and must not be treated as complete fixtures. A completed `manifest.json` plus a
successful verification is the handoff. Keep all generated files outside Git.
Do not regenerate published capture bytes in place.

One seeded Keras weight lineage per width supplies FP32 and fully INT8 exports
(four models total). Conversion uses the existing strict concrete-function path
so dilation is retained without relying on filename claims. Graph reports contain
actual operator names/versions, operand types, constant types and quantization.
INT8 graphs containing any floating-point tensor are rejected. Integer bias and
shape tensors remain integer; target accumulator precision and optimized kernel
dispatch are not inferred from graph types. This example does not export FP16 or
A16W8 and makes no native FP16 claim.

Calibration uses 32 deterministic uniform `[-1,1]` windows from seed+1. Held-out
cases are zero, constant -1, constant +1 and a deterministic multichannel sinusoid.
These limits describe the synthetic signal domain, not float32 numeric extremes.
Input hashes are checked for calibration/golden overlap. INT8 inputs use nearest
even rounding followed by saturation. The stored inputs are the exact bytes to
pass to a consumer; outputs are all 480 elements from the **exported** model using
LiteRT's single-thread `BUILTIN_REF` oracle. FP32 outputs additionally agree with
Keras at `rtol=1e-5, atol=1e-5`. INT8 goldens are not FP32 outputs cast to integers;
no quantized task accuracy or error budget is asserted.

The manifest records source-file hashes, Git head, Python and installed dependency
versions plus a dependency-list hash, recipe/configuration, seed, weight-array
hashes, artifact byte hashes, I/O quantization and graph reports. Retained weight
archives allow exact array restoration with `model.set_weights`; preserve the
recorded environment to reproduce conversion. A seed alone does not guarantee
identical bytes across dependency versions. Dependency hashes identify the version
list and repository lock, not hashes of installed binary distributions. Source
hashes describe the actual generator, including uncommitted changes if present.
The verifier assumes a trusted manifest; it detects corruption, checks metadata
against the model and replays goldens, but is not a signed supply-chain verifier.

Full output sizes are 480 bytes for INT8 and 1920 bytes for FP32. Consumers must
support those complete outputs; do not truncate to fit a transport limit. Board
admission, numerical comparison, timing, linked footprint, peak memory and energy
are separate measurements. No target measurements are made here.

Run the focused tests in the same environment:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. pytest -q tests/examples/test_compact_tcn.py
```

To reuse a retained generation during local tests, set `HELIA_TCN_FIXTURE` to its
absolute directory. Tests check topology and seeded weights, malformed recipes,
rounding/saturation, full golden replay, corruption rejection, float-fallback
rejection, and detection of an intentionally wrong dilation or golden output.
