"""Tests for S3 download utilities in helia_edge.utils.aws."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

from helia_edge.utils.aws import (
    download_s3_objects,
    download_s3_prefix,
    _list_s3_objects,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_s3_listing(prefix: str, files: list[str]) -> list[dict]:
    """Build a list of S3 object metadata dicts."""
    items = []
    for f in files:
        key = f"{prefix.rstrip('/')}/{f}"
        if key.endswith("/"):
            items.append({"Key": key, "Size": 0, "ETag": '""'})
        else:
            items.append({"Key": key, "Size": 100, "ETag": '"abc"'})
    return items


def _mock_client_for(items: list[dict]) -> MagicMock:
    """Create a mock S3 client that returns *items* from list_objects_v2."""
    client = MagicMock()
    client.list_objects_v2.return_value = {
        "Contents": items,
    }
    # download_file just touches the destination
    def fake_download(Bucket, Key, Filename):
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_text(Key)

    client.download_file.side_effect = fake_download
    return client


# ---------------------------------------------------------------------------
# download_s3_objects (deprecated) — verify behaviour preserved
# ---------------------------------------------------------------------------

def test_download_s3_objects_emits_deprecation_warning(tmp_path):
    """Old function must emit a DeprecationWarning."""
    items = _fake_s3_listing("myprefix", ["a.txt"])
    client = _mock_client_for(items)

    with warnings.catch_warnings(record=True) as w, \
         patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        warnings.simplefilter("always")
        download_s3_objects(
            bucket="b",
            prefix="myprefix",
            dst=tmp_path,
            progress=False,
        )
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    assert "download_s3_prefix" in str(dep_warnings[0].message)


def test_download_s3_objects_preserves_full_key(tmp_path):
    """Legacy function should still nest the full key (bug preserved)."""
    items = _fake_s3_listing("data", ["f1.h5", "f2.h5"])
    client = _mock_client_for(items)

    with warnings.catch_warnings(record=True), \
         patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        warnings.simplefilter("always")
        download_s3_objects(
            bucket="b",
            prefix="data",
            dst=tmp_path,
            progress=False,
        )
    # With the old bug, files land at dst / "data" / file
    assert (tmp_path / "data" / "f1.h5").exists()
    assert (tmp_path / "data" / "f2.h5").exists()


# ---------------------------------------------------------------------------
# download_s3_prefix — new correct function
# ---------------------------------------------------------------------------

def test_download_s3_prefix_strips_prefix(tmp_path):
    """Files should land directly under dst, not nested under the prefix."""
    items = _fake_s3_listing("datasets/ptbxl", ["00001.h5", "00002.h5"])
    client = _mock_client_for(items)

    with patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        n = download_s3_prefix(
            bucket="b",
            prefix="datasets/ptbxl",
            dst=tmp_path / "ptbxl",
            progress=False,
        )
    assert (tmp_path / "ptbxl" / "00001.h5").exists()
    assert (tmp_path / "ptbxl" / "00002.h5").exists()
    # Must NOT have the prefix duplicated
    assert not (tmp_path / "ptbxl" / "datasets").exists()
    assert n == 2


def test_download_s3_prefix_with_trailing_slash(tmp_path):
    """Prefix with trailing slash should work identically."""
    items = _fake_s3_listing("mydata", ["sub/a.bin"])
    client = _mock_client_for(items)

    with patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        download_s3_prefix(
            bucket="b",
            prefix="mydata/",
            dst=tmp_path,
            progress=False,
        )
    assert (tmp_path / "sub" / "a.bin").exists()


def test_download_s3_prefix_nested_structure(tmp_path):
    """Nested sub-directories under the prefix should be preserved."""
    items = _fake_s3_listing("root", ["a/b/c.txt", "a/d.txt", "e.txt"])
    client = _mock_client_for(items)

    with patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        download_s3_prefix(
            bucket="b",
            prefix="root",
            dst=tmp_path,
            progress=False,
        )
    assert (tmp_path / "a" / "b" / "c.txt").exists()
    assert (tmp_path / "a" / "d.txt").exists()
    assert (tmp_path / "e.txt").exists()


def test_download_s3_prefix_skips_directory_markers(tmp_path):
    """Keys ending with '/' are directory markers and should not create files."""
    items = [
        {"Key": "pfx/subdir/", "Size": 0, "ETag": '""'},
        {"Key": "pfx/subdir/file.txt", "Size": 42, "ETag": '"x"'},
    ]
    client = _mock_client_for(items)

    with patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        download_s3_prefix(
            bucket="b",
            prefix="pfx",
            dst=tmp_path,
            progress=False,
        )
    assert (tmp_path / "subdir").is_dir()
    assert (tmp_path / "subdir" / "file.txt").exists()


def test_download_s3_prefix_returns_count(tmp_path):
    """Return value should reflect how many files were actually downloaded."""
    items = _fake_s3_listing("p", ["a.txt", "b.txt", "c.txt"])
    client = _mock_client_for(items)

    with patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        n = download_s3_prefix(
            bucket="b",
            prefix="p",
            dst=tmp_path,
            progress=False,
        )
    assert n == 3


def test_download_s3_prefix_empty_listing(tmp_path):
    """An empty listing should succeed and return 0."""
    client = MagicMock()
    client.list_objects_v2.return_value = {"Contents": []}

    with patch("helia_edge.utils.aws._get_s3_client", return_value=client):
        n = download_s3_prefix(
            bucket="b",
            prefix="nothing",
            dst=tmp_path,
            progress=False,
        )
    assert n == 0


# ---------------------------------------------------------------------------
# _list_s3_objects pagination
# ---------------------------------------------------------------------------

def test_list_s3_objects_paginates():
    """Should follow NextContinuationToken until exhausted."""
    client = MagicMock()
    client.list_objects_v2.side_effect = [
        {
            "Contents": [{"Key": "p/a.txt", "Size": 1}],
            "NextContinuationToken": "tok1",
        },
        {
            "Contents": [{"Key": "p/b.txt", "Size": 2}],
        },
    ]
    items = _list_s3_objects(client, "bucket", "p")
    assert len(items) == 2
    assert items[0]["Key"] == "p/a.txt"
    assert items[1]["Key"] == "p/b.txt"
    assert client.list_objects_v2.call_count == 2
