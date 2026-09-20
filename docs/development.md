# Development

Python 3.12.3 is the minimum. TensorFlow is tested on 3.12–3.13 and Torch on
3.12–3.14. Python 3.15's native lazy-import syntax is outside that matrix;
`from __future__ import annotations` postpones annotations, not module imports.

## Package boundaries

- Helpers and sampling policies must import without Keras or a training backend.
- Portable tensor computation uses public Keras APIs. Gradient steps and export
  adapters may use their selected backend explicitly.
- KITs own dataset meaning, preprocessing policy and release criteria. Add shared
  abstractions only when real consumers establish the contract.
- Keep examples, benchmarks and test tools outside the installed package. Store
  run-specific measurements as artifacts, not source files.

Public lazy exports are declared once in adjacent `__init__.pyi` files.
`lazy-loader` uses those declarations at runtime; type checkers read the same
files. Package the stubs and `py.typed` marker when building distributions.
Keep import errors actionable and test base-only/TF-only/Torch-only environments.

Use enums for closed choices, dataclasses for internal records, and Pydantic for
validated external configuration. Named tuples suit structured tensor outputs
that must remain compatible with Keras trees and tuple unpacking. Dictionaries
remain appropriate for keyed collections and framework-defined serialization.
Keep comments about constraints or reasoning; put usage and architecture in docs.

## Static checking

```sh
uv sync --extra tensorflow --extra torch --extra litert --group typing
uv run ty check --error-on-warning
```

The dedicated typing environment contains both backends for resolving annotations.
Runtime isolation is tested separately. `ty` is pinned in the `typing` group,
included by `dev` and `ci`, and enforced by the CI typing job.

`tool.ty.src.include` lists the current gate: lazy public exports, modernized
sampling/helpers, patch layers, masked-autoencoder, examples and benchmark/test
tools. Static contract checks verify that public re-exports retain their types.
Legacy model families, converters and augmentation implementations still contain
typing debt and are outside this initial gate. There are no global rule
suppressions; extend coverage as those modules are corrected.

To inspect the remaining package debt, run `uv run ty check helia_edge`.
