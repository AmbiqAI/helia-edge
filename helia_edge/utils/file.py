"""
# File Utility API

This module provides utility functions to interact with files.

Functions:
    download_file: Download file from supplied url to destination streaming
    compute_checksum: Compute checksum of file
    load_pkl: Load pickled file
    save_pkl: Save python objects into pickle file
    resolve_template_path: Resolve templated path w/ supplied substitutions
"""

import gzip
import os
import hashlib
import pickle
from pathlib import Path
from string import Template
from typing import Any

import requests
from tqdm import tqdm


def download_file(src: str, dst: os.PathLike, progress: bool = True, chunk_size: int = 8192):
    """Download file from supplied url to destination streaming.

    Args:
        src (str): Source URL path
        dst (PathLike): Destination file path
        progress (bool, optional): Display progress bar. Defaults to True.

    """
    with requests.get(src, stream=True, timeout=3600 * 24) as r:
        r.raise_for_status()
        req_len = int(r.headers.get("Content-length", 0))
        prog_bar = tqdm(total=req_len, unit="iB", unit_scale=True) if progress else None
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                if prog_bar:
                    prog_bar.update(len(chunk))
            # END FOR
        # END WITH
    # END WITH


def compute_checksum(file: Path, checksum_type: str = "md5", chunk_size: int = 8192) -> str:
    """Compute checksum of file.

    Args:
        file (Path): File path
        checksum_type (str, optional): Checksum type. Defaults to "md5".
        chunk_size (int, optional): Chunk size. Defaults to 8192.

    Returns:
        str: Checksum value
    """
    if not file.is_file():
        raise FileNotFoundError(f"File {file} not found.")
    hash_algo = hashlib.new(checksum_type)
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_algo.update(chunk)
    return hash_algo.hexdigest()


def load_pkl(file: str, compress: bool = True) -> dict[str, Any]:
    """Load pickled file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): If file is compressed. Defaults to True.

    Returns:
        dict[str, Any]: Dictionary of pickled objects
    """
    if compress:
        with gzip.open(file, "rb") as fh:
            return pickle.load(fh)
    else:
        with open(file, "rb") as fh:
            return pickle.load(fh)


def save_pkl(file: str, compress: bool = True, **kwargs):
    """Save python objects into pickle file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): Whether to compress file. Defaults to True.
    """
    if compress:
        with gzip.open(file, "wb") as fh:
            pickle.dump(kwargs, fh, protocol=4)
    else:
        with open(file, "wb") as fh:
            pickle.dump(kwargs, fh, protocol=4)


def resolve_template_path(fpath: Path, **kwargs: Any) -> Path:
    """Resolve templated path w/ supplied substitutions.

    Args:
        fpath (Path): File path
        **kwargs (Any): Template arguments

    Returns:
        Path: Resolved file path
    """
    return Path(Template(str(fpath)).safe_substitute(**kwargs))
