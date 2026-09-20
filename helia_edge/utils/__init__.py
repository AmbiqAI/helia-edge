"""
# Utils API

The `utils` module provides utility functions to help with common tasks such as downloading files, setting random seeds, and exporting data.

## Available Utilities

- **[AWS](aws.md)**: Provides functions to download files from AWS S3.
- **[Environment](env.md)**: Provides functions to set up the environment and logging.
- **[Export](export.md)**: Provides functions to export data.
- **[Factory](factory.md)**: Provides functions to create objects.
- **[File](file.md)**: Provides functions to download and save files.
- **[Preprocessing](preprocessing.md)**: Provides functions to preprocess data.
- **[RNG](rng.md)**: Provides functions to set random seeds.
- **[Tensor](tensor.md)**: Provides functions to work with tensors.

"""

from helia_edge._lazy import lazy_exports

__getattr__, __dir__, __all__ = lazy_exports(
    __name__,
    {
        "download_s3_file": (".aws", "download_s3_file"),
        "download_s3_object": (".aws", "download_s3_object"),
        "download_s3_objects": (".aws", "download_s3_objects"),
        "download_s3_prefix": (".aws", "download_s3_prefix"),
        "env_flag": (".env", "env_flag"),
        "setup_logger": (".env", "setup_logger"),
        "silence_tensorflow": (".env", "silence_tensorflow"),
        "helia_export": (".export", "helia_export"),
        "ItemFactory": (".factory", "ItemFactory"),
        "create_factory": (".factory", "create_factory"),
        "download_file": (".file", "download_file"),
        "load_pkl": (".file", "load_pkl"),
        "save_pkl": (".file", "save_pkl"),
        "compute_checksum": (".file", "compute_checksum"),
        "resolve_template_path": (".file", "resolve_template_path"),
        "parse_factor": (".preprocessing", "parse_factor"),
        "convert_inputs_to_tf_dataset": (".preprocessing", "convert_inputs_to_tf_dataset"),
        "create_interleaved_dataset_from_generator": (".preprocessing", "create_interleaved_dataset_from_generator"),
        "create_dataset_from_data": (".preprocessing", "create_dataset_from_data"),
        "get_output_signature": (".preprocessing", "get_output_signature"),
        "get_output_signature_from_fn": (".preprocessing", "get_output_signature_from_fn"),
        "get_output_signature_from_gen": (".preprocessing", "get_output_signature_from_gen"),
        "set_random_seed": (".rng", "set_random_seed"),
        "uniform_id_generator": (".rng", "uniform_id_generator"),
        "random_id_generator": (".rng", "random_id_generator"),
        "matches_spec": (".tensor", "matches_spec"),
        "aws": (".aws", None),
        "env": (".env", None),
        "export": (".export", None),
        "factory": (".factory", None),
        "file": (".file", None),
        "preprocessing": (".preprocessing", None),
        "rng": (".rng", None),
        "tensor": (".tensor", None),
    },
)
