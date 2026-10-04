# Compact TCN models and exports

This example builds **seeded, untrained compact TCN models** and their LiteRT
exports for performance work. They are not task-quality models.

It builds `TcnParams` with `helia_edge.models.tcn.build`, using a preset fixed in `generate.py`:

- four small depthwise/pointwise blocks with batch normalization and ReLU6;
- dilations 1/2/4/8 and SE ratio 4;
- seed 20260925;
- widths 8 and 16, where each width replaces every block's `filters`;
- inputs `[1,240,14]` and outputs `[1,240,2]` logits.

Same padding and global SE pooling require whole windows. There is no causal,
streaming or recurrent-state claim. Benchmark recipes, reference inputs,
golden outputs and replay belong to the benchmark consumer, not this example.

Use Python 3.12 or 3.13 and the TensorFlow Keras backend with the repository's
`litert` extra. From a checkout, with its source on `PYTHONPATH`:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. python examples/compact_tcn/generate.py \
  --output /absolute/external/path/tcn-exports
```

The output directory must not already exist. A partial failure leaves a
partial directory, which must not be treated as complete; a written
`manifest.json` marks a complete generation. Keep generated files outside Git.

Each width has one seeded Keras weight lineage, which supplies an FP32 export
and a fully INT8 export (four models in total).

Before conversion, the generator validates the connected Keras graph against
the preset: SE pooling and gates, residual paths, pointwise kernels, dilations
and the linear output.

Conversion uses the existing strict concrete-function path, so dilation is
preserved. The emitted depthwise kernel, dilation, stride and padding options
are then checked in the exported graph. Operand types are restricted:

- FP32 exports may contain only FLOAT32 and INT32 operands;
- INT8 exports may contain only INT8 and INT32 operands, so any floating-point
  tensor is rejected.

Target accumulator precision and optimized kernel dispatch are not inferred
from graph types.

INT8 calibration uses 32 deterministic uniform `[-1,1]` windows from seed+1
(`--calibration-samples`, 1–256). On those windows, each FP32 export must agree
with Keras at `rtol=1e-5, atol=1e-5`, and the maximum error is recorded. The
windows are calibration data, not golden cases.

`manifest.json` records:

- source-file hashes, the Git head, Python and installed dependency versions;
- the preset and seed;
- calibration hashes and weight-array hashes;
- artifact byte hashes;
- I/O tensor details and quantization;
- graph reports and full output sizes (480 bytes INT8, 1920 bytes FP32).

Retained weight archives restore exact arrays with `model.set_weights`. A seed
alone does not guarantee identical bytes across dependency versions. New
output directories retain `LICENSE` and BSD-3-Clause source and
synthetic-weight provenance. Dependency versions are recorded for
reproducibility; they are not a dependency-license audit.

Run the focused tests in the same environment:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. pytest -q tests/examples/test_compact_tcn.py
```
