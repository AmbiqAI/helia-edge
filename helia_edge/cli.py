"""``helia-edge`` command line. Keras and TensorFlow are imported only by commands that need them."""

import importlib.util
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="heliaEDGE model export tools.")
export_app = typer.Typer(no_args_is_help=True, help="Create and reproduce exports with their records.")
app.add_typer(export_app, name="export")

DEFAULT_IO = {"fp32": "float32", "fp16": "float16", "a8w8": "int8", "a16w8": "int16"}


class Precision(StrEnum):
    """Precisions ``helia-edge export create`` exports."""

    FP32 = "fp32"
    FP16 = "fp16"
    A8W8 = "a8w8"
    A16W8 = "a16w8"


class Mode(StrEnum):
    """How ``helia-edge export create`` traces the model."""

    KERAS = "keras"
    CONCRETE = "concrete"


def _check_install(require_provenance: bool) -> None:
    from .export.result import environment_record, unidentified_install

    record = environment_record()
    if not record.identified:
        if require_provenance:
            typer.echo(f"error: {unidentified_install(record)}", err=True)
            raise typer.Exit(1)
        typer.echo(f"warning: {unidentified_install(record)}", err=True)


@export_app.command("create")
def export_create(
    spec: Annotated[Path, typer.Argument(help="ModelSpec file (YAML or JSON).", exists=True, dir_okay=False)],
    weights: Annotated[
        Path, typer.Option(help="A .weights.h5 file, or with --mapping the source file it imports.", exists=True)
    ],
    precision: Annotated[Precision, typer.Option(help="Numeric format of the export.")],
    out: Annotated[Path, typer.Option(help="Output directory.")],
    io_dtype: Annotated[str | None, typer.Option(help="I/O dtype; by default int8, int16 or the float type.")] = None,
    mapping: Annotated[str | None, typer.Option(help="Weight mapping of the spec's family to import with.")] = None,
    calibration: Annotated[Path | None, typer.Option(help="Calibration .npy for a8w8 and a16w8.", exists=True)] = None,
    resets: Annotated[list[int] | None, typer.Option(help="Calibration step that resets the state; repeat.")] = None,
    golden_inputs: Annotated[
        Path | None, typer.Option(help="Signal .npy for a golden@2 sequence.", exists=True)
    ] = None,
    golden_resets: Annotated[list[int] | None, typer.Option(help="Golden step that resets the state; repeat.")] = None,
    batch_size: Annotated[int, typer.Option(help="Batch of the exported model.", min=1, max=2**31 - 1)] = 1,
    mode: Annotated[Mode, typer.Option(help="How the model is traced.")] = Mode.KERAS,
    strict: Annotated[
        bool, typer.Option(help="For a8w8 and a16w8, refuse operators without an integer kernel.")
    ] = True,
    state_tie_tolerance: Annotated[
        float, typer.Option(help="Largest relative scale change when tying a state pair's scales.")
    ] = 0.01,
    require_provenance: Annotated[
        bool, typer.Option(help="Refuse unless helia-edge is a release or a git install at a commit.")
    ] = False,
) -> None:
    """Build SPEC, load its weights, export it and write model.tflite, model.weights.h5 and record.json.

    Exit 1 when the install does not identify its code under --require-provenance, or the export is refused.
    """
    if golden_resets and golden_inputs is None:
        typer.echo("error: --golden-resets needs --golden-inputs", err=True)
        raise typer.Exit(1)
    _check_install(require_provenance)
    try:
        from .export import ExportOptions, export
        from .export.api import _reference_build
        from .export.reproduce import family_mapping, imported_source, litert_unavailable, load_npy, load_spec
        from .importers import import_weights
        from .models.spec import build

        model_spec = load_spec(spec)
        unavailable = litert_unavailable()
        if unavailable:
            raise ValueError(unavailable[0])
        options = ExportOptions.model_validate(
            {"mode": mode.value, "strict": strict, "state_tie_tolerance": state_tie_tolerance}
        )
        with _reference_build():
            model = build(model_spec, batch_size=batch_size)
        weights_import = None
        if mapping is not None:
            weight_mapping = family_mapping(model_spec, mapping)
            import_weights(model, weight_mapping, weights)  # checks the file against the mapping's sha256
            weights_import = imported_source(weight_mapping, weight_mapping.source.sha256)
        else:
            model.load_weights(weights)
        result = export(
            model,
            precision=precision.value,
            io_dtype=DEFAULT_IO[precision.value] if io_dtype is None else io_dtype,
            calibration=None if calibration is None else load_npy(calibration),
            resets=resets or (),
            spec=model_spec,
            batch_size=batch_size,
            options=options,
            weights_import=weights_import,
        )
        if golden_inputs is not None:
            result = result.with_golden(load_npy(golden_inputs), golden_resets or ())
        path = result.write(out)
    except (OSError, ValueError, EOFError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"{result.record.artifact.file}: sha256 {result.record.artifact.sha256}")
    typer.echo(f"record: {path}")


@export_app.command("reproduce")
def export_reproduce(
    record: Annotated[Path, typer.Argument(help="record.json written by `export create` or export().")],
    weights: Annotated[Path, typer.Option(help="The weights file the record names (by digest or sha256).")],
    calibration: Annotated[Path | None, typer.Option(help="The calibration .npy the record names.")] = None,
    golden_inputs: Annotated[Path | None, typer.Option(help="The golden's input .npy, to compare the golden.")] = None,
    allow_env_mismatch: Annotated[
        bool, typer.Option(help="Compare even if the environment differs; differences are listed.")
    ] = False,
) -> None:
    """Export a record's model again and compare.

    Exit 0 same, 1 different, 2 environment differs or cannot export, 3 missing or mismatched input,
    4 the export failed (with its traceback).
    """
    import traceback

    from .export.reproduce import reproduce

    try:
        report = reproduce(record, weights, calibration, golden_inputs, allow_env_mismatch)
    except Exception:
        traceback.print_exc()
        typer.echo("error: the export failed, so nothing was compared", err=True)
        raise typer.Exit(4) from None
    for line in report.environment:
        typer.echo(f"environment: {line}")
    for line in report.differences:
        typer.echo(f"{'input' if report.status == 'input' else 'different'}: {line}")
    for line in report.skipped:
        typer.echo(f"not compared: {line}")
    typer.echo(report.status)
    raise typer.Exit({"same": 0, "different": 1, "environment": 2, "input": 3}[report.status])


@export_app.command("schema")
def export_schema() -> None:
    """Print the JSON Schema of the export record (helia-edge/export-record@1)."""
    from .export.record import ExportRecord

    typer.echo(json.dumps(ExportRecord.model_json_schema(by_alias=True), indent=1, sort_keys=True))


@app.command("inspect")
def inspect_model(
    model: Annotated[Path, typer.Argument(help="A .tflite model.", exists=True, dir_okay=False)],
) -> None:
    """Print a .tflite model's inputs, outputs and operators as JSON."""
    try:
        from .export.litert import operator_names, tensor_records
    except ModuleNotFoundError as exc:
        raise typer.BadParameter(
            f"inspect requires TensorFlow ({exc.name} is missing). Install helia-edge[litert]."
        ) from exc

    content = model.read_bytes()
    inputs, outputs = tensor_records(content)
    as_dict = lambda r: {**vars(r), "role": r.role.value, "dtype": r.dtype.value}  # noqa: E731
    report = {
        "inputs": [as_dict(r) for r in inputs],
        "outputs": [as_dict(r) for r in outputs],
        "operators": operator_names(content),
    }
    typer.echo(json.dumps(report, indent=1))


@app.command("info")
def info() -> None:
    """Print versions, the KERAS_BACKEND variable and installed optional capabilities."""
    from .export.result import environment_record

    record = environment_record()
    report = {
        "helia_edge": record.helia_edge,
        "helia_edge_commit": record.helia_edge_commit,
        "helia_edge_source": record.helia_edge_source,
        "python": record.python,
        "packages": dict(record.packages),
        "keras_backend_env": os.environ.get("KERAS_BACKEND"),
        "installed": {
            name: importlib.util.find_spec(module) is not None
            for name, module in (
                ("tensorflow", "tensorflow"),
                ("torch", "torch"),
                ("litert", "ai_edge_litert"),
                ("plotting", "matplotlib"),
                ("aws", "boto3"),
            )
        },
    }
    typer.echo(json.dumps(report, indent=1))


def main() -> None:
    """Entry point of the ``helia-edge`` command."""
    app()
