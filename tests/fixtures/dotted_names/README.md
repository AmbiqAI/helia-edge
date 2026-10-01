# Fixtures saved by earlier helia-edge versions

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

`two_outputs.keras` (outputs `out.a` and `out.b`, compiled with dict-keyed loss, metrics and loss
weights) and `tcn_layer_norm.keras` (a seeded TCN with `norm="layer"`, saved with Keras's
`LayerNormalization` over spatial axes) were saved by main `cb530275` on TensorFlow, with their
`*_io.npz` inputs, targets and outputs. Regenerate with `make_main_fixtures.py` from the same pull
request.

| File | SHA-256 |
| --- | --- |
| `two_outputs.keras` | `db762d30d24618d2f37d228a2aec9c0055eee174ac4b463b093fe830a5e7a6be` |
| `two_outputs_io.npz` | `1029c58c96365f7dc489b47244c2d8dbe27e9a6935d6894f69beddcfba60dc0a` |
| `tcn_layer_norm.keras` | `fb30957a198eca522c54ad27548fa254c404718d253c22337e28e16c92ba5e79` |
| `tcn_layer_norm_io.npz` | `2f1ec678d8734cd3414d20e3145d289bfccefe6b6135ab694441615c9edb59ff` |
