# :material-rocket-launch: Getting Started

## Install heliaEDGE

You can install __helia-edge__ from PyPI via:

```bash
pip install 'helia-edge[tensorflow]'
```

Alternatively, you can install using uv via:

```bash
uv add 'helia-edge[tensorflow]'
```

!!! note
    For Torch, install `helia-edge[torch]` and set `KERAS_BACKEND=torch` before
    importing Keras. Base `helia-edge` installs framework-independent helpers.
    See [backend support and model-loading migration](../backends.md) for tested
    components, optional export dependencies and limitations. JAX is not part of
    the tested support matrix.

Plotting and S3 access are optional. Select them alongside your backend when needed:

```sh
pip install 'helia-edge[tensorflow,plotting,aws]'
```

The training notebook uses plotting; S3 downloads and S3 model loading use `aws`.
Public plotting and download paths are unchanged. TensorFlow conversion and
custom-object registration do not require either extra.

## Requirements

* Python 3.12.3 or newer: 3.12–3.13 for TensorFlow; 3.12–3.14 for Torch/base.

Check the project's [pyproject.toml](https://github.com/AmbiqAI/helia-edge/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above will install all required dependencies.
