# Dotted layer names fixture

`efficientnetv2.keras` is a seeded, untrained EfficientNetV2 saved by helia-edge main `a677c85b` on the
TensorFlow backend. Its layer names contain `.` (for example `neck.conv` and `stage1_mbconv1_se_ex.act`),
as models saved by earlier helia-edge versions do. `efficientnetv2_io.npz` holds two inputs and the
TensorFlow outputs.

Regenerate with the script recorded in the pull request that added it (`make_dotted_fixture.py`), run with
`KERAS_BACKEND=tensorflow` and `PYTHONPATH` pointing at a checkout of `a677c85b`.

| File | SHA-256 |
| --- | --- |
| `efficientnetv2.keras` | `0d7c43e7dd69a270f7ca48c344fde8be3481b02c34d65c8171282e00e47985b3` |
| `efficientnetv2_io.npz` | `e45eb65e76c30ad07aad489baa8aefd57a4010144d8a92a31bba70bbe9ce8de9` |
