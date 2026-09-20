# Input-pipeline benchmark

`input_pipeline.py` is a developer tool for measuring synthetic preparation,
loading and training separately. It is not installed as part of `helia_edge` and
does not define a shared training or dataset API.

```sh
mkdir -p benchmarks/results
uv run --extra tensorflow python benchmarks/input_pipeline.py \
  --output benchmarks/results/local.json
```

Results are ignored by Git. Publish performance evidence as CI artifacts or with
the consuming project's experiment records. The synthetic workload checks timing
mechanics; it cannot establish throughput improvements for KIT datasets.

Numerical backend conformance is a test, maintained in
[`tests/compare_mae_backends.py`](../tests/compare_mae_backends.py).
