"""Versioned export recipe (``helia-edge/export@1``); importable without Keras."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .spec import CALIBRATED, ExportSpec

RECIPE_SCHEMA = "helia-edge/export@1"
SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NAME = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", max_length=64)]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UrlSource(_Strict):
    """A file fetched over HTTP(S) and checked against its sha256."""

    kind: Literal["url"]
    url: Annotated[str, Field(pattern=r"^https?://")]
    sha256: SHA256


class PathSource(_Strict):
    """A local file, relative to the recipe file unless absolute, checked against its sha256."""

    kind: Literal["path"]
    path: str
    sha256: SHA256


FileSource = Annotated[UrlSource | PathSource, Field(discriminator="kind")]


class ArraySource(_Strict):
    """A float32 array from a ``.npy`` file, or from key ``key`` of a ``.npz`` file."""

    kind: Literal["array"]
    file: FileSource
    key: str | None = None


class ParamsSeed(_Strict):
    """Build an architecture from params with seeded initialization; no trained weights."""

    kind: Literal["params_seed"]
    architecture: str
    params: dict[str, Any]
    input_shape: tuple[int, ...] | None = None
    num_classes: int | None = None
    seed: int


class ParamsWeights(_Strict):
    """Build an architecture from params, then load trained weights (``.weights.h5`` or ``.keras``)."""

    kind: Literal["params_weights"]
    architecture: str
    params: dict[str, Any]
    input_shape: tuple[int, ...] | None = None
    num_classes: int | None = None
    weights: FileSource


class KerasFile(_Strict):
    """Load a saved ``.keras`` model."""

    kind: Literal["keras_file"]
    file: FileSource


class TfliteImport(_Strict):
    """Use an existing ``.tflite`` model as is; the recipe then has no exports."""

    kind: Literal["tflite_import"]
    file: FileSource


ModelSource = Annotated[ParamsSeed | ParamsWeights | KerasFile | TfliteImport, Field(discriminator="kind")]


class CalibrationSpec(_Strict):
    """Calibration samples: the first ``samples`` rows in stored order, or all rows when None."""

    source: ArraySource
    samples: Annotated[int, Field(gt=0)] | None = None


class ReferenceSpec(_Strict):
    """Reference inputs; outputs are computed with LiteRT's reference kernels."""

    source: ArraySource
    samples: Annotated[int, Field(gt=0)] | None = None


class ExportEntry(ExportSpec):
    """One export: an ``ExportSpec`` with a name unique within the recipe."""

    name: NAME


class ExportRecipe(_Strict):
    """Everything needed to regenerate a set of exports. Every file carries a sha256."""

    schema_: Literal["helia-edge/export@1"] = Field(alias="schema")
    model: ModelSource
    calibration: CalibrationSpec | None = None
    reference: ReferenceSpec | None = None
    exports: tuple[ExportEntry, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _consistent(self) -> "ExportRecipe":
        names = [entry.name for entry in self.exports]
        if len(set(names)) != len(names):
            raise ValueError(f"export names must be unique: {names}")
        reserved = {"import", "manifest.json"} & set(names)
        if reserved:
            raise ValueError(f"export names {sorted(reserved)} are reserved")
        if isinstance(self.model, TfliteImport):
            if self.exports or self.calibration:
                raise ValueError("tflite_import recipes take no exports or calibration")
        elif not self.exports:
            raise ValueError("a recipe needs at least one export")
        calibrated = [entry.name for entry in self.exports if entry.precision in CALIBRATED]
        if calibrated and self.calibration is None:
            raise ValueError(f"exports {calibrated} need calibration")
        if self.calibration is not None and not calibrated:
            raise ValueError("calibration is given but no export is calibrated")
        return self


def load_recipe(path: Path) -> ExportRecipe:
    """Read a recipe from YAML (``.yaml``/``.yml``) or JSON."""
    import json

    text = Path(path).read_text()
    if Path(path).suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return ExportRecipe.model_validate(data)
