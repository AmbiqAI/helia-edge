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
- Model families: Keras-free `XParams` in `models/<family>_params.py` (strict, frozen, with a `family` field and
  `num_classes`), one `models/<family>.build(params, input_shape, *, batch_size=None, name=None)`, and
  `helia_edge.models.build(ModelSpec(params=..., input_shape=...))`. Add a family by adding it to `ModelParams` and
  the `match` in `models/spec.py`; no registries or `XModel` classes. MLPerf Tiny, FastEnhancer and Silero VAD take
  `input_shape=None` (fixed by their params).
- Export with `helia_edge.export.export_model` and run with `LiteRTRunner`, or `LiteRTStreamRunner` for streaming
  models with state pairs (export guide); the `converters` and `interpreters.tflite` classes are deprecated
  (they warn) and must not be used in new code or examples.
- Export recipes: `helia-edge export run RECIPE.yaml --out DIR` (`--require-provenance` for published exports),
  then `helia-edge export verify DIR/manifest.json` (exit 0 ok, 1 drift, 2 environment mismatch);
  `helia-edge export schema --kind recipe|manifest`.
- Extensions register in `helia_edge.registry` (exporters, architectures) or through the `helia_edge.plugins`
  entry-point group; custom train steps use `helia_edge.trainers.gradient_step` (MaskedAutoencoder does; the
  contrastive trainers are TensorFlow-only).
- Weights trained elsewhere load through `helia_edge.importers.import_weights` with a `WeightMapping` pinned to
  the source file's sha256 (extra `onnx` for ONNX), or a recipe's `params_import` model source; streaming
  recipes write `helia-model-zoo/golden@2` sequence goldens.
- Data: `helia_edge.data.to_grain` (extra `grain`) reads indexed records; `to_tf_dataset` and `to_torch_loader`
  wrap its batches. The tf.data generator helpers live in `helia_edge.data.tf_data` (re-exported by
  `helia_edge.utils`).
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
