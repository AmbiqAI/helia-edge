"""
# :simple-githubactions: Callbacks API

A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc). Beyond those provided by Keras, we provide a number of additional callbacks that can be used to monitor training, save models, and more.

## Available Callbacks

- **[TQDMProgressBar](./tqdm_progress_bar)**: Provides a progress bar for training.

Please check [Keras Callbacks](https://keras.io/api/callbacks/) for additional callbacks.

"""

from . import tqdm_progress_bar

from .tqdm_progress_bar import TQDMProgressBar
