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
partial directory: an export directory without `record.json` is incomplete,
and the generation is complete when all four records are written. Keep
generated files outside Git.

Each width has one seeded Keras weight lineage, which supplies an FP32 export
and a fully INT8 export (four models in total).

Before and after export, the generator validates the connected Keras graph
(the source model, and the model `export` rebuilds from the spec and converts) against
the preset: SE pooling and gates, residual paths, pointwise kernels, dilations
and the linear output.

Export uses the strict concrete-function path at batch 1, so dilation is
preserved. The emitted depthwise kernel, dilation, stride and padding options
are then checked in the exported graph. Operand types are restricted:

- FP32 exports may contain only FLOAT32 and INT32 operands;
- INT8 exports may contain only INT8 and INT32 operands, so any floating-point
  tensor is rejected.

Target accumulator precision and optimized kernel dispatch are not inferred
from graph types.

INT8 calibration uses 32 deterministic uniform `[-1,1]` windows from seed+1
(`--calibration-samples`, 1–256). On those windows, each FP32 export must agree
with Keras at `rtol=1e-5, atol=1e-5`, or the generation stops. The
windows are calibration data, not golden cases.

The output holds `calibration.npy`, `LICENSE` with `license.json` (BSD-3-Clause
source, synthetic seeded weights), and one directory per export
(`tcn-w8-fp32`, `tcn-w8-a8w8`, `tcn-w16-fp32`, `tcn-w16-a8w8`). Each holds the
export record written by `helia_edge.export.export`:

- `model.tflite` and `model.weights.h5`;
- `record.json`: the `ModelSpec`, the weights digest, the export settings, the
  calibration sha256, the artifact sha256, the I/O tensors with their quantization,
  and the environment (helia-edge install, Python, platform, package versions);
- `graph.json`, the example's graph report.

Reproduce any export with
`helia-edge export reproduce tcn-w8-a8w8/record.json --weights tcn-w8-a8w8/model.weights.h5 --calibration calibration.npy`
(no `--calibration` for FP32), in the environment that generated it. A seed
alone does not guarantee identical bytes across dependency versions; the record
names the Python, platform and package versions. It names the helia-edge code
only for a release or a git install at a commit
(`uv pip install 'helia-edge @ git+https://github.com/AmbiqAI/helia-edge@<commit>'`);
run from a checkout on `PYTHONPATH`, it records an `unknown` install and the
generator warns.

Run the focused tests in the same environment:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. pytest -q tests/examples/test_compact_tcn.py
```
