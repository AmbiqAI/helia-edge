"""
# Model utilities API

This module provides utility functions to work with Keras models.

Functions:
    make_divisible: Ensure layer has # channels divisble by divisor
    load_model: Loads a Keras model stored either remotely or locally
    append_layers: Appends layers to a model by cloning it and adding the layers

"""

import os
import tempfile
import itertools
import glob
from pathlib import Path
import keras

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


def load_model(model_path: os.PathLike) -> keras.Model:
    """Loads a Keras model stored either remotely or locally.
    NOTE: Currently supports wandb, s3, and https for remote.

    Args:
        model_path (str): Source path
            WANDB: wandb:[[entity/]project/]collectionName:[alias]
            FILE: file:/path/to/model.tf
            S3: s3:bucket/prefix/model.tf
            https: https://path/to/model.tf

    Returns:
        keras.Model: Model
    """

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
                model = keras.models.load_model(model_path)
            # END WITH

        case "s3":
            import boto3  # pylint: disable=C0415
            from botocore import UNSIGNED  # pylint: disable=C0415
            from botocore.client import Config  # pylint: disable=C0415

            session = boto3.Session()
            client = session.client("s3", config=Config(signature_version=UNSIGNED))
            model_path = model_path.removeprefix("s3:")
            path_parts = model_path.split(":")[1].split("/")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model_ext = Path(model_path).suffix
                dst_path = Path(tmpdirname) / f"model{model_ext}"
                client.download_file(
                    Bucket=path_parts[0],
                    Key="/".join(path_parts[1:]),
                    Filename=str(dst_path),
                )
                model = keras.models.load_model(dst_path)
            # END WITH

        case "https":
            with tempfile.TemporaryDirectory() as tmpdirname:
                model_ext = Path(model_path).suffix
                dst_path = Path(tmpdirname) / f"model{model_ext}"
                download_file(model_path, dst_path)
                model = keras.models.load_model(dst_path)
            # END WITH

        case _:
            model_path = model_path.removeprefix("file:")
            model = keras.models.load_model(model_path)
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
