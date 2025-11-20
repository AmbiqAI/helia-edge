"""AWS Cloud Utility API

This module provides utility functions to interact with AWS services.

Functions:
    download_s3_file: Download a file from S3
    download_s3_object: Download an object from S3
    download_s3_objects: Download all objects in a S3 bucket with a given prefix


"""

import os
import functools
from pathlib import Path
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
    """Download all objects in a S3 bucket with a given prefix

    Args:
        bucket (str): Bucket name
        prefix (str): Prefix to filter objects
        dst (Path): Destination directory
        checksum (str, optional): Checksum type. Defaults to "size".
        progress (bool, optional): Show progress bar. Defaults to True.
        num_workers (int | None, optional): Number of workers. Defaults to None.
        config (Config | None, optional): Boto3 config. Defaults to Config(signature_version=UNSIGNED).

    """

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
