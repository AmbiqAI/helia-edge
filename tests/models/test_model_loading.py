"""Regression tests for S3 model loading and optional AWS guidance."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import keras
import numpy as np
import pytest
from botocore import UNSIGNED

from helia_edge.models import load_model


@pytest.fixture()
def saved_model(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, np.ndarray, np.ndarray]:
    suffix = request.param
    model = keras.Sequential(
        [
            keras.Input(shape=(2,)),
            keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.layers[0].set_weights(
        [
            np.asarray([[2.0], [-1.0]], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
        ]
    )
    x = np.asarray([[1.0, 3.0], [-2.0, 4.0]], dtype=np.float32)
    expected = keras.ops.convert_to_numpy(model(x, training=False))
    path = tmp_path / f"fixture{suffix}"
    model.save(path)
    return path, x, expected


@pytest.mark.parametrize("saved_model", [".keras", ".h5"], indirect=True)
@pytest.mark.parametrize("prefix", ["s3:", "s3://", "S3://"])
def test_load_model_s3_forms_download_and_reload_real_model(
    saved_model: tuple[Path, np.ndarray, np.ndarray], prefix: str
) -> None:
    source, x, expected = saved_model
    key = f"models/nested/revision:v1{source.suffix}"
    session = MagicMock()
    client = MagicMock()

    def download_file(*, Bucket: str, Key: str, Filename: str) -> None:
        shutil.copyfile(source, Filename)

    client.download_file.side_effect = download_file
    session.client.return_value = client
    with patch("boto3.Session", return_value=session):
        loaded = load_model(f"{prefix}bucket/{key}")

    session.client.assert_called_once()
    assert session.client.call_args.kwargs["config"].signature_version == UNSIGNED
    client.download_file.assert_called_once()
    assert client.download_file.call_args.kwargs["Bucket"] == "bucket"
    assert client.download_file.call_args.kwargs["Key"] == key
    np.testing.assert_allclose(keras.ops.convert_to_numpy(loaded(x, training=False)), expected)


@pytest.mark.parametrize(
    "model_path", ["s3:", "s3://", "s3://bucket", "s3:bucket/", "s3:/key.keras", "s3:///key.keras"]
)
def test_load_model_s3_rejects_empty_bucket_or_key_before_sdk(model_path: str) -> None:
    with patch("boto3.Session") as session:
        with pytest.raises(ValueError, match="non-empty bucket and key"):
            load_model(model_path)
    session.assert_not_called()


def test_load_model_s3_guides_when_aws_is_missing() -> None:
    with patch.dict(sys.modules, {"boto3": None}):
        with pytest.raises(ImportError, match=r"Install helia-edge\[aws\]"):
            load_model("s3:bucket/model.keras")
