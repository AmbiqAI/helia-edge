"""AWS Cloud Utility API

This module provides utility functions to interact with AWS services.

Functions:
    download_s3_file: Download a file from S3
    download_s3_object: Download an object from S3
    download_s3_prefix: Download all objects under an S3 prefix into a local directory
    download_s3_objects: Download all objects in a S3 bucket with a given prefix (deprecated)


"""

import os
import warnings
import functools
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

from .env import setup_logger
from .file import compute_checksum

logger = setup_logger(__name__)


def _get_s3_client(config: Config | None = None) -> boto3.client:
    """Get S3 client

    Args:
        config (Config | None, optional): Boto3 config. Defaults to None.

    Returns:
        boto3.client: S3 client
    """
    session = boto3.Session()
    return session.client("s3", config=config)


def download_s3_file(
    key: str,
    dst: Path,
    bucket: str,
    client: boto3.client = None,
    checksum: str = "size",
    config: Config | None = Config(signature_version=UNSIGNED),
) -> bool:
    """Download a file from S3

    Args:
        key (str): Object key
        dst (Path): Destination path
        bucket (str): Bucket name
        client (boto3.client): S3 client
        checksum (str, optional): Checksum type. Defaults to "size".
        config (Config, optional): Boto3 config. Defaults to Config(signature_version=UNSIGNED).

    Returns:
        bool: True if file was downloaded, False if already exists
    """

    if client is None:
        client = _get_s3_client(config)

    if not dst.is_file():
        pass
    elif checksum == "size":
        obj = client.head_object(Bucket=bucket, Key=key)
        if dst.stat().st_size == obj["ContentLength"]:
            return False
    elif checksum == "md5":
        obj = client.head_object(Bucket=bucket, Key=key)
        etag = obj["ETag"]
        checksum_type = obj.get("ChecksumAlgorithm", ["md5"])[0]
        calculated_checksum = compute_checksum(dst, checksum)
        if etag == calculated_checksum and checksum_type.lower() == "md5":
            return False
    # END IF

    client.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(dst),
    )

    return True


def download_s3_object(
    item: dict[str, str],
    dst: Path,
    bucket: str,
    client: boto3.client = None,
    checksum: str = "size",
    config: Config | None = Config(signature_version=UNSIGNED),
) -> bool:
    """Download an object from S3

    Args:
        item (dict[str, str]): Object metadata
        dst (Path): Destination path
        bucket (str): Bucket name
        client (boto3.client): S3 client
        checksum (str, optional): Checksum type. Defaults to "size".
        config (Config, optional): Boto3 config. Defaults to Config(signature_version=UNSIGNED).

    Returns:
        bool: True if file was downloaded, False if already exists
    """

    # Is a directory, skip
    if item["Key"].endswith("/"):
        os.makedirs(dst, exist_ok=True)
        return False

    if not dst.is_file():
        pass
    elif checksum == "size":
        if dst.stat().st_size == item["Size"]:
            return False
    elif checksum == "md5":
        etag = item["ETag"]
        checksum_type = item.get("ChecksumAlgorithm", ["md5"])[0]
        calculated_checksum = compute_checksum(dst, checksum)
        if etag == calculated_checksum and checksum_type.lower() == "md5":
            return False
    # END IF

    if client is None:
        client = _get_s3_client()

    os.makedirs(dst.parent, exist_ok=True)

    client.download_file(
        Bucket=bucket,
        Key=item["Key"],
        Filename=str(dst),
    )

    return True


def download_s3_objects(
    bucket: str,
    prefix: str,
    dst: Path,
    checksum: str = "size",
    progress: bool = True,
    num_workers: int | None = None,
    config: Config | None = Config(signature_version=UNSIGNED),
):
    """Download all objects in a S3 bucket with a given prefix.

    .. deprecated::
        Use :func:`download_s3_prefix` instead.  This function preserves the
        full S3 key (including the prefix) when building local paths, which
        causes files to be nested one level too deep when ``dst`` already
        contains the prefix directory.  The replacement strips the prefix so
        that ``dst`` is always the root of the downloaded tree.

    Args:
        bucket (str): Bucket name
        prefix (str): Prefix to filter objects
        dst (Path): Destination directory
        checksum (str, optional): Checksum type. Defaults to "size".
        progress (bool, optional): Show progress bar. Defaults to True.
        num_workers (int | None, optional): Number of workers. Defaults to None.
        config (Config | None, optional): Boto3 config. Defaults to Config(signature_version=UNSIGNED).

    """

    warnings.warn(
        "download_s3_objects is deprecated and has a known path-nesting bug. "
        "Use download_s3_prefix instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    client = _get_s3_client(config)

    # Fetch all objects in the bucket with the given prefix
    items = []
    fetching = True
    next_token = None
    while fetching:
        if next_token is None:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        else:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=next_token)
        items.extend(response["Contents"])
        next_token = response.get("NextContinuationToken", None)
        fetching = next_token is not None
    # END WHILE

    logger.debug(f"Found {len(items)} objects in {bucket}/{prefix}")

    os.makedirs(dst, exist_ok=True)

    func = functools.partial(download_s3_object, bucket=bucket, client=client, checksum=checksum)

    pbar = tqdm(total=len(items), unit="objects") if progress else None

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = (
            executor.submit(
                func,
                item,
                dst / item["Key"],
            )
            for item in items
        )
        for future in as_completed(futures):
            err = future.exception()
            if err:
                logger.exception("Failed on file")
            if pbar:
                pbar.update(1)
        # END FOR
    # END WITH


def _list_s3_objects(
    client: boto3.client,
    bucket: str,
    prefix: str,
) -> list[dict]:
    """Paginate through all objects under *prefix* in *bucket*.

    Args:
        client (boto3.client): S3 client.
        bucket (str): Bucket name.
        prefix (str): Key prefix.

    Returns:
        list[dict]: Object metadata dicts from ``list_objects_v2``.
    """
    items: list[dict] = []
    next_token = None
    while True:
        kwargs: dict = {"Bucket": bucket, "Prefix": prefix}
        if next_token is not None:
            kwargs["ContinuationToken"] = next_token
        response = client.list_objects_v2(**kwargs)
        items.extend(response.get("Contents", []))
        next_token = response.get("NextContinuationToken")
        if next_token is None:
            break
    return items


def download_s3_prefix(
    bucket: str,
    prefix: str,
    dst: Path,
    checksum: str = "size",
    progress: bool = True,
    num_workers: int | None = None,
    config: Config | None = Config(signature_version=UNSIGNED),
) -> int:
    """Download all objects under an S3 prefix into a local directory.

    Unlike :func:`download_s3_objects`, this function **strips the prefix**
    from each object key before joining it with *dst*, so that *dst* becomes
    the root of the downloaded tree.

    Example::

        # S3 objects:  s3://my-bucket/datasets/ptbxl/00001.h5
        #              s3://my-bucket/datasets/ptbxl/00002.h5
        download_s3_prefix(
            bucket="my-bucket",
            prefix="datasets/ptbxl",
            dst=Path("./data/ptbxl"),
        )
        # Results in: ./data/ptbxl/00001.h5
        #             ./data/ptbxl/00002.h5

    Args:
        bucket (str): Bucket name.
        prefix (str): Key prefix to filter objects.  A trailing ``/`` is
            added automatically if missing.
        dst (Path): Local directory that will mirror the contents found
            under *prefix*.
        checksum (str, optional): Checksum strategy (``"size"`` or
            ``"md5"``). Defaults to ``"size"``.
        progress (bool, optional): Show a ``tqdm`` progress bar.
            Defaults to ``True``.
        num_workers (int | None, optional): Thread-pool size.  ``None``
            uses the :class:`~concurrent.futures.ThreadPoolExecutor`
            default.
        config (Config | None, optional): Boto3 client config.
            Defaults to unsigned requests.

    Returns:
        int: Number of objects downloaded (excludes skipped / up-to-date).
    """

    # Normalise prefix so stripping is reliable.
    norm_prefix = prefix.rstrip("/") + "/" if prefix and not prefix.endswith("/") else prefix

    client = _get_s3_client(config)
    items = _list_s3_objects(client, bucket, prefix)

    logger.debug(f"Found {len(items)} objects in s3://{bucket}/{prefix}")

    os.makedirs(dst, exist_ok=True)

    func = functools.partial(download_s3_object, bucket=bucket, client=client, checksum=checksum)
    downloaded = 0

    pbar = tqdm(total=len(items), unit="files") if progress else None

    def _local_path(key: str) -> Path:
        """Strip the prefix and join onto *dst*."""
        relative = key[len(norm_prefix):] if key.startswith(norm_prefix) else key
        return dst / PurePosixPath(relative)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(func, item, _local_path(item["Key"])): item
            for item in items
        }
        for future in as_completed(futures):
            err = future.exception()
            if err:
                item = futures[future]
                logger.exception("Failed downloading %s", item["Key"])
            else:
                if future.result():
                    downloaded += 1
            if pbar:
                pbar.update(1)
        # END FOR
    # END WITH

    if pbar:
        pbar.close()

    return downloaded
