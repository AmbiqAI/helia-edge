# helia-edge contributor notes

Keras 3 add-on for training and exporting models to Ambiq edge targets. Full guide:
`astro-site/src/content/docs/guide/development.mdx`.

## Environment and checks

- Python 3.12.3+. TensorFlow extras exist for 3.12-3.13 only; Torch for 3.12-3.14.
- Full environment: `uv sync --locked --all-extras --group ci`.
- Hooks (prek, `.pre-commit-config.yaml`); CI runs the same stages:
  - `uv run prek run --all-files --hook-stage pre-commit`: ruff format, ruff check, `uv lock --check`.
  - `uv run prek run --all-files --hook-stage pre-push`: `ty check --error-on-warning`, fast pytest set.
- ruff, ty and prek are pinned exactly in the `ci` group; change a pin in `pyproject.toml` and relock.
- Export recipes: `helia-edge export run RECIPE.yaml --out DIR`, then `helia-edge export verify DIR/manifest.json`
  (exit 0 ok, 1 drift, 2 environment mismatch); `helia-edge export schema --kind recipe|manifest`.
- Extensions register in `helia_edge.registry` (exporters, architectures) or through the `helia_edge.plugins`
  entry-point group; custom train steps use `helia_edge.trainers.gradient_step` (MaskedAutoencoder does; the
  contrastive trainers are TensorFlow-only).
- Backend tests: set `KERAS_BACKEND` (`tensorflow` or `torch`) before importing Keras; see the
  `backend` job in `.github/workflows/ci.yaml` for each backend's test set.

## Rules

- Base installs must import without Keras, a backend, plotting or AWS extras; public exports are
  declared in adjacent `__init__.pyi` files and loaded lazily.
- Do not rename or remove Keras-registered classes (`helia_export`); saved models reference them.
- Enums for closed choices, dataclasses for internal records, pydantic for validated external
  configuration.
- No trained weights or datasets in the repository.
- Conventional commit subjects; releases are cut by release-please (minor versions while below 1.0).
