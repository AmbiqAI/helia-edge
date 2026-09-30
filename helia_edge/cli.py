"""``helia-edge`` command line. Keras and TensorFlow are imported only by commands that need them."""

import importlib.util
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="heliaEDGE model export tools.")
export_app = typer.Typer(no_args_is_help=True, help="Run, verify and describe export recipes.")
app.add_typer(export_app, name="export")


class SchemaKind(StrEnum):
    RECIPE = "recipe"
    MANIFEST = "manifest"


@export_app.command("run")
def export_run(
    recipe: Annotated[Path, typer.Argument(help="Recipe file (YAML or JSON).", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("exports"),
    only: Annotated[list[str] | None, typer.Option(help="Export name to run; repeat for several.")] = None,
) -> None:
    """Regenerate a recipe's exports and write OUT/manifest.json."""
    from .export.run import run_recipe

    manifest = run_recipe(recipe, out, only=only)
    for entry in manifest.entries:
        typer.echo(f"{entry.name}: {entry.model.path} sha256 {entry.model.sha256}")
    typer.echo(f"manifest: {Path(out) / 'manifest.json'}")


@export_app.command("verify")
def export_verify(
    manifest: Annotated[Path, typer.Argument(help="manifest.json written by `export run`.")],
    allow_env_mismatch: Annotated[
        bool, typer.Option(help="Regenerate even if the environment differs; differences are listed.")
    ] = False,
) -> None:
    """Regenerate a manifest's exports and compare them.

    Exit 0 ok, 1 drift, 2 environment mismatch, 3 missing or invalid manifest.
    """
    import pydantic

    from .export.run import verify_manifest

    try:
        report = verify_manifest(manifest, allow_env_mismatch=allow_env_mismatch)
    except (FileNotFoundError, pydantic.ValidationError, ValueError) as exc:
        typer.echo(f"invalid manifest {manifest}: {exc}", err=True)
        raise typer.Exit(3) from exc
    for line in report.environment_differences:
        typer.echo(f"environment: {line}")
    for line in report.differences:
        typer.echo(f"drift: {line}")
    typer.echo(report.status)
    raise typer.Exit({"ok": 0, "drift": 1, "env_mismatch": 2}[report.status])


@export_app.command("schema")
def export_schema(
    kind: Annotated[SchemaKind, typer.Option(help="Which document to describe.")] = SchemaKind.RECIPE,
) -> None:
    """Print the JSON Schema of an export recipe or manifest."""
    from .export.manifest import ExportManifest
    from .export.recipe import ExportRecipe

    model = ExportRecipe if kind == SchemaKind.RECIPE else ExportManifest
    typer.echo(json.dumps(model.model_json_schema(by_alias=True), indent=1, sort_keys=True))


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
    app()
