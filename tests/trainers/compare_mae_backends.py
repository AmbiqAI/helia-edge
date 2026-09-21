"""Compare fixed-input forward/gradient/update evidence from isolated backends."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorflow-python")
    parser.add_argument("--torch-python")
    parser.add_argument("--tensorflow-evidence", type=Path)
    parser.add_argument("--torch-evidence", type=Path)
    args = parser.parse_args()
    if not ((args.tensorflow_python and args.torch_python) or (args.tensorflow_evidence and args.torch_evidence)):
        parser.error("supply both Python environments or both evidence files")
    fixture = Path(__file__).with_name("mae_parity_fixture.py")
    with tempfile.TemporaryDirectory() as directory:
        paths = {}
        for backend in ("tensorflow", "torch"):
            if getattr(args, f"{backend}_evidence"):
                paths[backend] = getattr(args, f"{backend}_evidence")
                continue
            paths[backend] = Path(directory) / f"{backend}.npz"
            env = dict(os.environ, KERAS_BACKEND=backend, CUDA_VISIBLE_DEVICES="-1")
            subprocess.run(
                [getattr(args, f"{backend}_python"), str(fixture), "--output", str(paths[backend])],
                env=env,
                check=True,
                timeout=120,
            )
        with np.load(paths["tensorflow"]) as left, np.load(paths["torch"]) as right:
            assert set(left.files) == set(right.files)
            errors = {}
            for name in left.files:
                np.testing.assert_allclose(left[name], right[name], rtol=1e-5, atol=1e-6, err_msg=name)
                errors[name] = float(np.max(np.abs(left[name] - right[name])))
        print(json.dumps({"rtol": 1e-5, "atol": 1e-6, "maximum_absolute_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
