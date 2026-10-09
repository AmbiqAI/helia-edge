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
run from a checkout (on `PYTHONPATH`, or installed editable) it records an
`unknown` or `local` install and the generator warns. Two such checkouts record
the same environment, so `reproduce` cannot tell their code apart.

## Identified install and normal CLI export

For a code-identified record, use a checkout at a **published commit** and
install that commit in an external environment. From the checkout root:

```sh
TCN_COMMIT="$(git rev-parse HEAD)"
TCN_ENV=/absolute/external/path/tcn-env
export TCN_OUT=/absolute/external/path/tcn-exports
uv venv --python 3.12 "$TCN_ENV"
uv pip install --python "$TCN_ENV/bin/python" \
  "helia-edge[litert] @ git+https://github.com/AmbiqAI/helia-edge@$TCN_COMMIT"
export KERAS_BACKEND=tensorflow
env -u PYTHONPATH "$TCN_ENV/bin/python" "$PWD/examples/compact_tcn/generate.py" \
  --output "$TCN_OUT"
```

Keep using this environment with `PYTHONPATH` unset, outside the checkout when
running inline Python, so the identified install supplies `helia_edge`.
`record.json` records the resolved dependency versions; the install command
alone does not pin them. Reproduction requires that recorded environment.

The record's `model` field is already a normal `ModelSpec`. Extract it, then
create the same width-8 A8W8 artifact with the ordinary CLI and saved weights:

```sh
cd "$TCN_OUT"
env -u PYTHONPATH "$TCN_ENV/bin/python" - <<'PY'
import json
from pathlib import Path

record = json.loads(Path("tcn-w8-a8w8/record.json").read_text())
Path("spec.json").write_text(json.dumps(record["model"], indent=2) + "\n")
PY
env -u PYTHONPATH "$TCN_ENV/bin/helia-edge" export create spec.json \
  --weights tcn-w8-a8w8/model.weights.h5 --calibration calibration.npy \
  --precision a8w8 --batch-size 1 --mode concrete --require-provenance \
  --out cli-create
env -u PYTHONPATH "$TCN_ENV/bin/helia-edge" inspect cli-create/model.tflite
env -u PYTHONPATH "$TCN_ENV/bin/helia-edge" export reproduce cli-create/record.json \
  --weights cli-create/model.weights.h5 --calibration calibration.npy
```

Use a new `cli-create` directory. Successful reproduction prints `same` and
returns zero; environment or artifact differences must be investigated.
Pass the standard `record.json`, model, saved weights, calibration and license
files to consumers. Tensor order, names, shapes, dtypes and quantizers are in
the record. This model has one signal input, one signal output and no state
pairs; consumers need no cache initialization or reset schedule.

## Complete-input host smoke

From the output directory above, invoke the original A8W8 export once with
the first synthetic calibration window. The maintained runner encodes all
3360 input elements with the actual scale and zero point, then returns all
480 raw INT8 output elements. Save both arrays outside Git:

```sh
env -u PYTHONPATH "$TCN_ENV/bin/python" - <<'PY'
from pathlib import Path

import numpy as np
from helia_edge.export import LiteRTRunner

runner = LiteRTRunner(
    Path("tcn-w8-a8w8/model.tflite").read_bytes(), reference_kernels=True
)
fed = runner.encode(np.load("calibration.npy", allow_pickle=False)[:1])
outputs = runner.run(fed)
assert fed.shape == (1, 240, 14) and fed.dtype == np.int8
assert outputs.shape == (1, 240, 2) and outputs.dtype == np.int8
np.save("host-input.npy", fed, allow_pickle=False)
np.save("host-output.npy", outputs, allow_pickle=False)
print("Invoke passed:", fed.shape, "->", outputs.shape)
PY
```

This checks complete host invocation and output extents. It establishes neither
task accuracy nor target execution, kernel selection, memory fit or timing.
The saved output is a smoke result, not a task-qualified golden reference.

Return to the checkout root to run the focused tests in the same environment:

```sh
KERAS_BACKEND=tensorflow PYTHONPATH=. pytest -q tests/examples/test_compact_tcn.py
```
