"""
# Model utilities API

This module provides utility functions to work with Keras models.

Functions:
    make_divisible: Ensure layer has # channels divisble by divisor
    load_model: Loads a Keras model stored either remotely or locally
    append_layers: Appends layers to a model by cloning it and adding the layers

"""

import glob
import itertools
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import keras

from .._lazy import Extra, optional_imports
from ..utils import download_file


def make_divisible(v: int, divisor: int = 4, min_value: int | None = None) -> int:
    """Ensure layer has # channels divisble by divisor
       https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    Args:
        v (int): Number of channels
        divisor (int, optional): Divisor. Defaults to 4.
        min_value (int | None, optional): Min # channels. Defaults to None.

    Returns:
        int: Number of channels
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def undot_layer_names(config: Any) -> tuple[Any, dict[str, str]]:
    """Rename layers whose names contain ``.`` in a saved Keras model config.

    Torch registers each layer as a module attribute, and attribute names cannot contain ``.``.
    Each ``.`` becomes ``_``. Layer names, connections (``keras_history``) and the model's input
    and output lists are renamed together; weights are stored by structure, not by name, so they
    need no change.

    Args:
        config: The parsed ``config.json`` of a ``.keras`` file.

    Returns:
        tuple: The renamed config (a new object) and the mapping from old to new names.

    Raises:
        ValueError: If a new name equals another layer's name.
    """
    names: set[str] = set()

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            if "class_name" in node and isinstance(node.get("config"), dict):
                name = node["config"].get("name")
                if isinstance(name, str):
                    names.add(name)
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(config)
    mapping = {name: name.replace(".", "_") for name in sorted(names) if "." in name}
    targets: dict[str, str] = {}
    for old, new in mapping.items():
        if new in names or new in targets:
            other = new if new in names else targets[new]
            raise ValueError(f"Cannot rename layer {old!r} to {new!r}: it would collide with {other!r}")
        targets[new] = old

    def rename(node: Any, key: str | None = None) -> Any:
        if isinstance(node, dict):
            out = {k: rename(v, k) for k, v in node.items()}
            if "class_name" in out and isinstance(out.get("config"), dict) and out["config"].get("name") in mapping:
                out["config"] = {**out["config"], "name": mapping[out["config"]["name"]]}
            if "class_name" in out and out.get("name") in mapping:
                out["name"] = mapping[out["name"]]
            return out
        if isinstance(node, list):
            if key == "keras_history" and node and node[0] in mapping:
                return [mapping[node[0]], *node[1:]]
            if key in ("input_layers", "output_layers"):
                return rename_reference(node)
            return [rename(value) for value in node]
        return node

    def rename_reference(reference: Any) -> Any:
        if isinstance(reference, list) and reference and isinstance(reference[0], str):
            return [mapping.get(reference[0], reference[0]), *reference[1:]]
        if isinstance(reference, list):
            return [rename_reference(value) for value in reference]
        return reference

    return rename(config), mapping


def _load_keras_file(path: os.PathLike) -> keras.Model:
    """Load a local model file; on Torch, ``.keras`` files with dotted layer names are renamed first."""
    path = Path(path)
    if keras.backend.backend() != "torch" or path.suffix != ".keras":
        return keras.models.load_model(path)
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        renamed, mapping = undot_layer_names(config)
        if not mapping:
            return keras.models.load_model(path)
        with tempfile.TemporaryDirectory() as tmpdirname:
            target = Path(tmpdirname) / path.name
            with zipfile.ZipFile(target, "w") as out:
                for item in archive.infolist():
                    data = json.dumps(renamed).encode() if item.filename == "config.json" else archive.read(item)
                    out.writestr(item, data)
            return keras.models.load_model(target)


def load_model(model_path: os.PathLike) -> keras.Model:
    """Loads a Keras model stored either remotely or locally.
    NOTE: Currently supports wandb, s3, and https for remote.

    Args:
        model_path (str): Source path
            WANDB: wandb:[[entity/]project/]collectionName:[alias]
            FILE: file:/path/to/model.tf
            S3: s3:bucket/prefix/model.tf
            https: https://path/to/model.tf

    On the Torch backend, layer names containing ``.`` (common in models saved by earlier
    helia-edge versions) are renamed to use ``_`` before loading; see ``undot_layer_names``.

    Returns:
        keras.Model: Model
    """

    from .._serialization import register_keras_serializables

    register_keras_serializables()
    model_path = str(model_path)
    model_prefix: str = model_path.split(":")[0].lower() if ":" in model_path else ""

    match model_prefix:
        case "wandb":
            import wandb  # pylint: disable=C0415

            api = wandb.Api()
            model_path = model_path.removeprefix("wandb:")
            artifact = api.artifact(model_path, type="model")
            with tempfile.TemporaryDirectory() as tmpdirname:
                artifact.download(tmpdirname)
                model_path = tmpdirname
                # Find the model file
                file_paths = [glob.glob(f"{tmpdirname}/*.{f}") for f in ["keras", "tf", "h5"]]
                file_paths = list(itertools.chain.from_iterable(file_paths))
                if not file_paths:
                    raise FileNotFoundError("Model file not found in artifact")
                model_path = file_paths[0]
                model = _load_keras_file(model_path)
            # END WITH

        case "s3":
            s3_path = model_path.partition(":")[2]
            if s3_path.startswith("//"):
                s3_path = s3_path[2:]
            bucket, separator, key = s3_path.partition("/")
            if not separator or not bucket or not key:
                raise ValueError("S3 model path must contain a non-empty bucket and key")

            with optional_imports(Extra.AWS):
                import boto3
                from botocore import UNSIGNED
                from botocore.client import Config

            session = boto3.Session()
            client = session.client("s3", config=Config(signature_version=UNSIGNED))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model_ext = Path(key).suffix
                dst_path = Path(tmpdirname) / f"model{model_ext}"
                client.download_file(
                    Bucket=bucket,
                    Key=key,
                    Filename=str(dst_path),
                )
                model = _load_keras_file(dst_path)
            # END WITH

        case "https":
            with tempfile.TemporaryDirectory() as tmpdirname:
                model_ext = Path(model_path).suffix
                dst_path = Path(tmpdirname) / f"model{model_ext}"
                download_file(model_path, dst_path)
                model = _load_keras_file(dst_path)
            # END WITH

        case _:
            model_path = model_path.removeprefix("file:")
            model = _load_keras_file(model_path)
    # END MATCH

    return model


def append_layers(model: keras.Model, layers: list[keras.Layer], copy_weights: bool = True) -> keras.Model:
    """Appends layers to a model by cloning it and adding the layers.

    Args:
        model (keras.Model): Model
        layers (list[keras.layers.Layer]): Layers to append
        copy_weights (bool, optional): Copy weights. Defaults to True.

    Returns:
        keras.Model: Model
    """

    last_layer_name = model.layers[-1].name

    def call_function(layer, *args, **kwargs):
        out = layer(*args, **kwargs)
        if layer.name == last_layer_name:
            for new_layer in layers:
                out = new_layer(out)
            # END FOR
        # END IF
        return out

    # END DEF

    model_clone = keras.models.clone_model(model, call_function=call_function)
    if copy_weights:
        model_clone.set_weights(model.get_weights())
    return model_clone
