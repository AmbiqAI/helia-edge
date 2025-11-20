"""
# C++ Converter Utilities

This module provides utility functions to convert numpy arrays and binary files to C++ format.

Functions:
    array_dump: Generate C array of values from flattened numpy array.
    xxd_c_dump: Generate C array of values from binary file.

"""

import os

import numpy as np
import numpy.typing as npt


def array_dump(
    data: npt.NDArray,
    dst_path: os.PathLike,
    var_name: str = "test_stimulus",
    var_dtype: str | None = None,
    row_len: int = 12,
    is_header: bool = False,
):
    """Generate C array of values from flattened numpy array.

    Args:
        data (npt.NDArray): Data array
        dst_path (PathLike): C file destination path
        var_name (str, optional): C variable name. Defaults to "test_stimulus".
        var_dtype (str | None, optional): C variable type. Defaults to None.
        row_len (int, optional): Elements to write per row. Defaults to 12.
        is_header (bool): Write as header or source C file. Defaults to source.

    Example:

    ```python
    import helia_edge as helia

    data = np.random.rand(10, 10).astype(np.float32)
    dst_path = "test_stimulus.h"
    helia.converters.cpp.array_dump(
        data,
        dst_path,
        var_name="test_stimulus",
        var_dtype="float32_t",
        row_len=12,
        is_header=True
    )
    ```
    """
    data = data.flatten()

    if var_dtype is not None:
        pass
    elif isinstance(data[0], np.floating):
        var_dtype = "float"
    elif isinstance(data[0], np.int8):
        var_dtype = "int8_t"
    elif isinstance(data[0], np.int16):
        var_dtype = "int16_t"
    elif isinstance(data[0], np.integer):
        var_dtype = "int32_t"
    else:
        raise ValueError("Unsupported dtype")

    with open(dst_path, "w", encoding="UTF-8") as wfp:
        if is_header:
            wfp.write(f"#ifndef __{var_name.upper()}_H{os.linesep}")
            wfp.write(f"#define __{var_name.upper()}_H{os.linesep}")

        wfp.write(f"#include <cstdint>{os.linesep}{os.linesep}")

        wfp.write(f"const {var_dtype} {var_name}[] = {{{os.linesep}")
        for row in range(0, len(data), row_len):
            wfp.write("  " + ", ".join((str(val) for val in data[row : row + row_len])) + f", {os.linesep}")
        # END FOR
        wfp.write(f"}};{os.linesep}")
        wfp.write(f"const unsigned int {var_name}_len = {len(data)};{os.linesep}")
        if is_header:
            wfp.write(f"#endif // __{var_name.upper()}_H{os.linesep}")
    # END WITH


def xxd_c_dump(
    src_path: os.PathLike,
    dst_path: os.PathLike,
    var_name: str = "model",
    chunk_len: int = 12,
    is_header: bool = False,
):
    """Generate C like char array of hex values from binary source. Equivalent to `xxd -i src_path > dst_path`
        but with added features to provide # columns and variable name.

    Args:
        src_path (PathLike): Binary file source path
        dst_path (PathLike): C file destination path
        var_name (str, optional): C variable name. Defaults to 'model'.
        chunk_len (int, optional): Number of elements per row. Defaults to 12.
        is_header (bool): Write as header or source C file. Defaults to source.

    Example:

    ```python
    import helia_edge as helia

    src_path = "model.tflite"
    dst_path = "model.h"
    helia.converters.cpp.xxd_c_dump(
        src_path,
        dst_path,
        var_name="model",
        chunk_len=12,
        is_header=True
    )
    ```
    """
    var_len = 0
    with open(src_path, "rb", encoding=None) as rfp, open(dst_path, "w", encoding="UTF-8") as wfp:
        if is_header:
            wfp.write(f"#ifndef __{var_name.upper()}_H{os.linesep}")
            wfp.write(f"#define __{var_name.upper()}_H{os.linesep}")

        wfp.write(f"const unsigned char {var_name}[] = {{{os.linesep}")
        for chunk in iter(lambda: rfp.read(chunk_len), b""):
            wfp.write("  " + ", ".join((f"0x{c:02x}" for c in chunk)) + f", {os.linesep}")
            var_len += len(chunk)
        # END FOR
        wfp.write(f"}};{os.linesep}")
        wfp.write(f"const unsigned int {var_name}_len = {var_len};{os.linesep}")
        if is_header:
            wfp.write(f"#endif // __{var_name.upper()}_H{os.linesep}")
    # END WITH
