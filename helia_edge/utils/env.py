"""
# Environment Utility API

This module provides utility functions to interact with the environment.

Functions:
    setup_logger: Setup logger with Rich
    env_flag: Return the specified environment variable coerced to a bool
    silence_tensorflow: Silence every unnecessary warning from tensorflow
    disable_tensorflow_gpu: Disable TensorFlow GPU
"""

import os
import sys
from contextlib import contextmanager
import logging

from rich.logging import RichHandler


def setup_logger(
    log_name: str,
    level: int | None = None,
    recursive: bool = True,
    file_path: os.PathLike | None = None,
) -> logging.Logger:
    """Setup logger with Rich

    Args:
        log_name (str): Logger name

    Returns:
        logging.Logger: Logger
    """
    root_name = log_name.split(".")[0] if recursive else log_name
    new_logger = logging.getLogger(root_name)
    needs_init = not new_logger.handlers

    match level:
        case 0:
            log_level = logging.ERROR
        case 1:
            log_level = logging.INFO
        case 2 | 3 | 4:
            log_level = logging.DEBUG
        case None:
            log_level = None
        case _:
            log_level = logging.INFO
    # END MATCH

    if needs_init:
        logging.basicConfig(force=True, handlers=[RichHandler(rich_tracebacks=True)])
        new_logger.propagate = False
        new_logger.handlers = [RichHandler(show_time=False)]

    if log_level is not None:
        new_logger.setLevel(log_level)

    if file_path and not any(isinstance(handler, logging.FileHandler) for handler in new_logger.handlers):
        file_handler = logging.FileHandler(file_path, mode="w")
        file_handler.setLevel(log_level)
        new_logger.addHandler(file_handler)

    return new_logger


def env_flag(env_var: str, default: bool = False) -> bool:
    """Return the specified environment variable coerced to a bool, as follows:
    - When the variable is unset, or set to the empty string, return `default`.
    - When the variable is set to a truthy value, returns `True`.
      These are the truthy values:
          - 1
          - true, yes, on
    - When the variable is set to the anything else, returns False.
       Example falsy values:
          - 0
          - no
    - Ignore case and leading/trailing whitespace.

    Args:
        env_var (str): Environment variable name
        default (bool, optional): Default value. Defaults to False.

    Returns:
        bool: Value of environment variable
    """
    environ_string = os.environ.get(env_var, "").strip().lower()
    if not environ_string:
        return default
    return environ_string in ["1", "true", "yes", "on"]


def silence_tensorflow():
    """Silence every unnecessary warning from tensorflow."""
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["AUTOGRAPH_VERBOSITY"] = "5"
    # We wrap this inside a try-except block
    # because we do not want to be the one package
    # that crashes when TensorFlow is not installed
    # when we are the only package that requires it
    # in a given Jupyter Notebook, such as when the
    # package import is simply copy-pasted.
    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(3)
    except ModuleNotFoundError:
        pass


def disable_tensorflow_gpu():
    """Disable TensorFlow GPU"""
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], "GPU")

@contextmanager
def suppress_os_stdio():
    """
    Redirects the OS stdout (fd=1) and stderr (fd=2) to /dev/null, then restores them on exit.
    Suppresses all output **unless** an exception is raised, in which case it will be logged.

    Example:
        with suppress_os_stdio():
            noisy_function()
    """
    # 1) Flush Python buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # 2) Duplicate original FDs so we can restore them later
    orig_stdout_fd = os.dup(sys.stdout.fileno())
    orig_stderr_fd = os.dup(sys.stderr.fileno())

    try:
        # 3) Open /dev/null and redirect
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull_fd, sys.stdout.fileno())
        os.dup2(devnull_fd, sys.stderr.fileno())
        os.close(devnull_fd)

        # 4) Enter user’s block
        yield

    except Exception:
        # 5) An error occurred! Log it (the FDs are still /dev/null at this moment).
        logging.exception("Error inside suppressed stdout/stderr block")
        raise

    finally:
        # 6) Restore original FDs no matter what
        os.dup2(orig_stdout_fd, sys.stdout.fileno())
        os.dup2(orig_stderr_fd, sys.stderr.fileno())
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)
