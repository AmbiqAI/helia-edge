# Changelog

## [0.7.0](https://github.com/AmbiqAI/helia-edge/compare/v0.6.2...v0.7.0) (2026-10-04)


### ⚠ BREAKING CHANGES

* **importers:** WeightMapping.format moved to SourcePin.format. The importers and weight_mappings registries are removed from helia_edge.registry, so plugins can no longer add readers or recipe mappings. load_fastenhancer_weights, fastenhancer_t_onnx_weights and FASTENHANCER_T_ONNX_NAMES are removed; use import_weights with FASTENHANCER_T_ONNX or fastenhancer_mapping. fastenhancer_weight_shapes moved to helia_edge.models.fastenhancer_params.
* **models:** removed:
    - the XModel classes of these families and silero_vad_v6();
    - the mlperf_tiny_{kws,vww,resnet,ad} builders;
    - XParams.name;
    - get_config/from_config and the Keras registration of the CorNET,
      TimePPG, MiniResNet and FastEnhancer params.
    Also breaking:
    - models are named after their family and FastEnhancer's state is
      renamed;
    - unet_layer and unext_layer take num_classes from the params;
    - UNet and UNext params reject unknown fields;
    - block Params classes moved to the *_params modules;
    - Silero builds with a dynamic batch unless batch_size=1.
* **models:** the XModel classes (model_from_params, layer_from_params) of these families, XParams.name, dict params and TcnParams.get_config/from_config with its Keras registration are removed; models are named after their family.
* models built with the layer helpers (TCN with squeeze-excite, EfficientNetV2, MBConv, RegNet, UNet with layer norm, Conformer) export LiteRT files with new tensor names, so their sha256 changes once; shapes, dtypes and outputs are unchanged and saved weights load unchanged. Models with layer norm over spatial axes now save helia_edge>LayerNormalization, so plain keras.saving.load_model needs helia_edge.register_keras_serializables() first.
* ResNet and RegNet blocks with a stride above 1 only on an axis of size 1, at constant width, now build differently. ResNet and RegNet y blocks project the shortcut, so weights saved before no longer load into a rebuilt model. RegNet z blocks omit the residual Add; old weights still load but give different outputs. All other configurations that built before are unchanged.
* helia_edge.optimizers, helia_edge.quantizers, helia_edge.converters.torch, helia_edge.utils.train and helia_edge.models.cct are removed.
* TfLiteKerasConverter.convert raises for INT8 and INT16X8 without test_x instead of calibrating on random data, and debug_quantization() after a float conversion without test_x raises.
* unify preprocessing augmentation hooks

### Features

* add a typed LiteRT export API ([#56](https://github.com/AmbiqAI/helia-edge/issues/56)) ([a5831ad](https://github.com/AmbiqAI/helia-edge/commit/a5831adc6435a3da27742fda88387d1a41a491c6)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* add CorNET heart-rate regressor constructor ([#46](https://github.com/AmbiqAI/helia-edge/issues/46)) ([fbe0986](https://github.com/AmbiqAI/helia-edge/commit/fbe0986a77423ee41df3a881e104f025c0afb98b)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* add export recipes, manifests and the helia-edge command ([#59](https://github.com/AmbiqAI/helia-edge/issues/59)) ([d5b6ae3](https://github.com/AmbiqAI/helia-edge/commit/d5b6ae33b0a581c209d56db1e8ec78b3490b4d0b)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* add faithful MLPerf Tiny reference constructors ([0df0b6e](https://github.com/AmbiqAI/helia-edge/commit/0df0b6e77e1f53f7ac482df0707bd23b20c25b42)), closes [#33](https://github.com/AmbiqAI/helia-edge/issues/33)
* add faithful MLPerf Tiny reference constructors ([#34](https://github.com/AmbiqAI/helia-edge/issues/34)) ([4aa2bd9](https://github.com/AmbiqAI/helia-edge/commit/4aa2bd9c5e1292ee7fb1224765c323774e3dff28)), closes [#33](https://github.com/AmbiqAI/helia-edge/issues/33)
* add FastEnhancer folded-inference streaming model ([#44](https://github.com/AmbiqAI/helia-edge/issues/44)) ([ca85f6e](https://github.com/AmbiqAI/helia-edge/commit/ca85f6e10d03e85b5b0384c052f3d62dc9ac68ff)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* add native FP16 LiteRT conversion ([#52](https://github.com/AmbiqAI/helia-edge/issues/52)) ([e908d25](https://github.com/AmbiqAI/helia-edge/commit/e908d25b6b622ffcb90cae55ec089b11a73a7d3d)), closes [#51](https://github.com/AmbiqAI/helia-edge/issues/51)
* add registries and backend-dispatched training steps ([#60](https://github.com/AmbiqAI/helia-edge/issues/60)) ([d611ee2](https://github.com/AmbiqAI/helia-edge/commit/d611ee2ffe45dca4a8af5285658adc9c9536608b)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* add seeded compact TCN models and exports ([#35](https://github.com/AmbiqAI/helia-edge/issues/35)) ([134e118](https://github.com/AmbiqAI/helia-edge/commit/134e11899cfbc394cf3c6cfa37bff63e9c3d7355)), closes [#32](https://github.com/AmbiqAI/helia-edge/issues/32)
* add the data layer with Grain, tf.data and Torch adapters ([#63](https://github.com/AmbiqAI/helia-edge/issues/63)) ([a342b10](https://github.com/AmbiqAI/helia-edge/commit/a342b10797f7a898124f614943611941b0526dff)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* add TimePPG heart-rate regressor constructor ([#45](https://github.com/AmbiqAI/helia-edge/issues/45)) ([747e885](https://github.com/AmbiqAI/helia-edge/commit/747e885c62c909339c4223f2f1ba2740a3ed8ad0)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* add typed MiniResNet-v1 architecture ([97276b2](https://github.com/AmbiqAI/helia-edge/commit/97276b248acca67cfb3d752cec5193252f36fdf7)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* add typed public architecture APIs ([#37](https://github.com/AmbiqAI/helia-edge/issues/37)) ([0e29c97](https://github.com/AmbiqAI/helia-edge/commit/0e29c97627cf8ff62a6454efb94a0798611e0b59)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* **export:** record how helia-edge was installed in export manifests ([#89](https://github.com/AmbiqAI/helia-edge/issues/89)) ([8546dec](https://github.com/AmbiqAI/helia-edge/commit/8546dec39624f058e7c8e6bb399b98288d817753)), closes [#85](https://github.com/AmbiqAI/helia-edge/issues/85)
* import weights through checked mappings, with Silero VAD v6 as the first model ([#84](https://github.com/AmbiqAI/helia-edge/issues/84)) ([007cb8e](https://github.com/AmbiqAI/helia-edge/commit/007cb8ea4183216fe36e48d29404747c9dcaa619)), closes [#83](https://github.com/AmbiqAI/helia-edge/issues/83) [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* isolate backends and define typed training contracts ([db22394](https://github.com/AmbiqAI/helia-edge/commit/db223944df561a284049229935e43f9f2f5e8dff))
* register Silero VAD v6 and export streaming models with golden@2 sequences ([#86](https://github.com/AmbiqAI/helia-edge/issues/86)) ([eb3ab90](https://github.com/AmbiqAI/helia-edge/commit/eb3ab90a9cc1a57ba8a9791e95fd58c4bf3e5c37)), closes [#85](https://github.com/AmbiqAI/helia-edge/issues/85) [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* serialize validated TCN architecture configs ([72553b1](https://github.com/AmbiqAI/helia-edge/commit/72553b1e618090f776edee5da10b867970294257)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* stream LSTM state as explicit model I/O and tie INT state pairs at export ([#82](https://github.com/AmbiqAI/helia-edge/issues/82)) ([6827688](https://github.com/AmbiqAI/helia-edge/commit/682768845c4d8ce8d8f4a269a24e663d25ec996d)), closes [#81](https://github.com/AmbiqAI/helia-edge/issues/81) [#53](https://github.com/AmbiqAI/helia-edge/issues/53)


### Bug Fixes

* allow disabled squeeze excitation in small TCN blocks ([1e1a076](https://github.com/AmbiqAI/helia-edge/commit/1e1a076e5678dd9cef03ed8498ee208271acb6bc)), closes [#36](https://github.com/AmbiqAI/helia-edge/issues/36)
* build and load models on Torch: layer names and layer norm ([#80](https://github.com/AmbiqAI/helia-edge/issues/80)) ([a63bb6e](https://github.com/AmbiqAI/helia-edge/commit/a63bb6ea083393419ba8b5df89a4bac04867f79d)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53) [#73](https://github.com/AmbiqAI/helia-edge/issues/73) [#74](https://github.com/AmbiqAI/helia-edge/issues/74)
* build the model families that failed to construct ([#79](https://github.com/AmbiqAI/helia-edge/issues/79)) ([cb53027](https://github.com/AmbiqAI/helia-edge/commit/cb530275a691d76229e3c8a00070351f3c12a069)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53) [#66](https://github.com/AmbiqAI/helia-edge/issues/66) [#67](https://github.com/AmbiqAI/helia-edge/issues/67) [#68](https://github.com/AmbiqAI/helia-edge/issues/68) [#69](https://github.com/AmbiqAI/helia-edge/issues/69) [#70](https://github.com/AmbiqAI/helia-edge/issues/70) [#71](https://github.com/AmbiqAI/helia-edge/issues/71) [#72](https://github.com/AmbiqAI/helia-edge/issues/72)
* pin MLPerf layout and discriminate fixture outputs ([2c95757](https://github.com/AmbiqAI/helia-edge/commit/2c957576bbc6f57d77be55b2dc3e5988b799b9bb)), closes [#33](https://github.com/AmbiqAI/helia-edge/issues/33)
* prune orphan tensors and unused opcodes in native FP16 graphs ([#55](https://github.com/AmbiqAI/helia-edge/issues/55)) ([3698661](https://github.com/AmbiqAI/helia-edge/commit/3698661225016bd7b9cc6cec7f94790cc688e10f)), closes [#51](https://github.com/AmbiqAI/helia-edge/issues/51) [#53](https://github.com/AmbiqAI/helia-edge/issues/53)


### Documentation

* adopt shared mobile navigation and compact terminals ([#62](https://github.com/AmbiqAI/helia-edge/issues/62)) ([74a5656](https://github.com/AmbiqAI/helia-edge/commit/74a5656ce3a350d0636712294c7d6a7a81b50685))
* correct Apache license name spacing ([ce1de70](https://github.com/AmbiqAI/helia-edge/commit/ce1de7048f06675185dd6dbb7199d4e743e4a180)), closes [#33](https://github.com/AmbiqAI/helia-edge/issues/33)
* migrate heliaEDGE site to Astro with generated API reference ([#49](https://github.com/AmbiqAI/helia-edge/issues/49)) ([ca4c781](https://github.com/AmbiqAI/helia-edge/commit/ca4c781df3af7375fbb24a221ff5de1b27163283))
* refine heliaEDGE onboarding and progressive examples ([#50](https://github.com/AmbiqAI/helia-edge/issues/50)) ([c31306a](https://github.com/AmbiqAI/helia-edge/commit/c31306a5953bdb14874997bff9347ec683f60e74))
* remove the trained checkpoint bundled with the CIFAR guide ([#48](https://github.com/AmbiqAI/helia-edge/issues/48)) ([2b8c461](https://github.com/AmbiqAI/helia-edge/commit/2b8c46101c587993b1119e007cb9ca6320c3ffcd)), closes [#47](https://github.com/AmbiqAI/helia-edge/issues/47)


### Code Refactoring

* **importers:** mappings live with their family; FastEnhancer on WeightMapping ([#95](https://github.com/AmbiqAI/helia-edge/issues/95)) ([dc5253e](https://github.com/AmbiqAI/helia-edge/commit/dc5253eeb4b1978dcbfd2b9a78f433c7bed9b612)), closes [#91](https://github.com/AmbiqAI/helia-edge/issues/91)
* **models:** ModelSpec and one build per classifier family ([#92](https://github.com/AmbiqAI/helia-edge/issues/92)) ([bd026af](https://github.com/AmbiqAI/helia-edge/commit/bd026af62eed3dde880a62d52a6038ccf6a908e3)), closes [#91](https://github.com/AmbiqAI/helia-edge/issues/91)
* **models:** ModelSpec builds for the remaining families ([#94](https://github.com/AmbiqAI/helia-edge/issues/94)) ([22363c2](https://github.com/AmbiqAI/helia-edge/commit/22363c280eca40e8bf86f9c263547a43f5ffaf69)), closes [#91](https://github.com/AmbiqAI/helia-edge/issues/91)
* remove empty modules and MkDocs, deprecate the legacy converters ([#65](https://github.com/AmbiqAI/helia-edge/issues/65)) ([851fc25](https://github.com/AmbiqAI/helia-edge/commit/851fc2533f2b8c84fe7cdc28123f671e1183fc05)), closes [#53](https://github.com/AmbiqAI/helia-edge/issues/53)
* unify preprocessing augmentation hooks ([60931e8](https://github.com/AmbiqAI/helia-edge/commit/60931e87ca6892e7d2c8962b4b04d62689b9819f)), closes [#38](https://github.com/AmbiqAI/helia-edge/issues/38)

## [0.6.2](https://github.com/AmbiqAI/helia-edge/compare/v0.6.1...v0.6.2) (2026-04-23)


### Bug Fixes

* add actions:write permission for release workflow dispatch ([96df439](https://github.com/AmbiqAI/helia-edge/commit/96df439ec092b7a4ac75205d13360f2dac9f176e))

## [0.6.1](https://github.com/AmbiqAI/helia-edge/compare/v0.6.0...v0.6.1) (2026-04-22)


### Bug Fixes

* chain PyPI publish in release-please workflow ([#20](https://github.com/AmbiqAI/helia-edge/issues/20)) ([6989a20](https://github.com/AmbiqAI/helia-edge/commit/6989a205fe34e71b3caaf41672aadc7190763133))

## [0.6.0](https://github.com/AmbiqAI/helia-edge/compare/v0.5.0...v0.6.0) (2026-04-22)


### Features

* add EmaResidualVectorQuantizer with EMA codebook updates ([#17](https://github.com/AmbiqAI/helia-edge/issues/17)) ([85a0dde](https://github.com/AmbiqAI/helia-edge/commit/85a0dde68bc2085f73c1e62b9d16bf71fc10a1a0))
* add EmaResidualVectorQuantizer with EMA codebook updates ([#17](https://github.com/AmbiqAI/helia-edge/issues/17)) ([85a0dde](https://github.com/AmbiqAI/helia-edge/commit/85a0dde68bc2085f73c1e62b9d16bf71fc10a1a0))


### Bug Fixes

* add download_s3_prefix, deprecate download_s3_objects ([#18](https://github.com/AmbiqAI/helia-edge/issues/18)) ([a53bebe](https://github.com/AmbiqAI/helia-edge/commit/a53bebe83787325a151a8814d61085a7f64ddd6a))
* add download_s3_prefix, deprecate download_s3_objects ([#18](https://github.com/AmbiqAI/helia-edge/issues/18)) ([a53bebe](https://github.com/AmbiqAI/helia-edge/commit/a53bebe83787325a151a8814d61085a7f64ddd6a))

## [0.5.0](https://github.com/AmbiqAI/helia-edge/compare/v0.4.1...v0.5.0) (2026-04-03)


### Features

* Add initial pytest. ([96026e6](https://github.com/AmbiqAI/helia-edge/commit/96026e60d783744d074cf5f8f37ce435fa77aa22))
* Add initial pytest. ([8cf3a32](https://github.com/AmbiqAI/helia-edge/commit/8cf3a32c6e9e2c676a08a3d16d466428508924e3))
* add PRD metric and refresh docs ([#12](https://github.com/AmbiqAI/helia-edge/issues/12)) ([0a9592f](https://github.com/AmbiqAI/helia-edge/commit/0a9592fe417d61fa7f9f67d211fc650ddf1578de))


### Bug Fixes

* add from_config() to VQAutoencoder and align get_config() serialization ([#11](https://github.com/AmbiqAI/helia-edge/issues/11)) ([a56c467](https://github.com/AmbiqAI/helia-edge/commit/a56c467453cf440a606eb2e921219c82f31f7bda))
* VQAutoencoder Keras 3 compatibility ([#10](https://github.com/AmbiqAI/helia-edge/issues/10)) ([a617d28](https://github.com/AmbiqAI/helia-edge/commit/a617d2856d761e85280782784d05ccff0aab1496))

## [0.4.1](https://github.com/AmbiqAI/helia-edge/compare/v0.4.0...v0.4.1) (2025-12-16)


### Bug Fixes

* Update build-system configuration and add tool.uv settings ([58e7d93](https://github.com/AmbiqAI/helia-edge/commit/58e7d93dc9b99bcb93a81384fa53ad467d0bdab9))
* Update build-system configuration and add tool.uv settings ([82744ed](https://github.com/AmbiqAI/helia-edge/commit/82744ed789fb46ab74769f5b0254d8d179a2cfc9))

## [0.4.0](https://github.com/AmbiqAI/helia-edge/compare/v0.3.0...v0.4.0) (2025-12-12)


### Features

* Add compression techniques ([#4](https://github.com/AmbiqAI/helia-edge/issues/4)) ([5754c8f](https://github.com/AmbiqAI/helia-edge/commit/5754c8faab9039253212f63d1ee6381c5edf4d24))

## [0.3.0](https://github.com/AmbiqAI/helia-edge/compare/v0.2.2...v0.3.0) (2025-11-20)


### Features

* Add confusion metric w/ norm. ([524c65b](https://github.com/AmbiqAI/helia-edge/commit/524c65b06cd259fbaf90370597c220f51ba11e8c))
* Add confusion metric w/ norm. ([4ccd0d3](https://github.com/AmbiqAI/helia-edge/commit/4ccd0d34f95d4ea148a3e843b21e14cf69a6b365))
* Add confusion metric w/ norm. ([959d76b](https://github.com/AmbiqAI/helia-edge/commit/959d76b114edf5662a201da0c1022b2467352a4b))
* Add devcontainer. ([be90f44](https://github.com/AmbiqAI/helia-edge/commit/be90f443c712cf67765046aecca8cad8f60e685a))
* Add evaluate to contrastive trainer. ([54f3d58](https://github.com/AmbiqAI/helia-edge/commit/54f3d58febde19c0e91bc3f91c7b3066e66cfc53))
* Add fast conformer and metaformer. ([7e1aea5](https://github.com/AmbiqAI/helia-edge/commit/7e1aea5aeffa1a7c52d59d328a062523fc3e7686))
* Add file handler. ([ba2f434](https://github.com/AmbiqAI/helia-edge/commit/ba2f43484bf44509f0634b0c13f869d8c2227135))
* Add force training mode to pipelines. Useful in tf.data pipeline when want to apply augmentations regardless. ([0781fd4](https://github.com/AmbiqAI/helia-edge/commit/0781fd4a64110c55d358dd3f885f52e8e91a41a0))
* Add Freq mixstyle. ([2b288ca](https://github.com/AmbiqAI/helia-edge/commit/2b288ca06d14f2bbf2b19ef7b5a66bdfa358ce0b))
* Add generic factory class. ([26dd95d](https://github.com/AmbiqAI/helia-edge/commit/26dd95dcc47058fb02799851936b813de841453f))
* Add LayerNorm preprocessing. ([0176ba4](https://github.com/AmbiqAI/helia-edge/commit/0176ba4704a87434aee8877c7d7410efb9e384eb))
* Add RandomChannel augmentation. ([0270adf](https://github.com/AmbiqAI/helia-edge/commit/0270adf43bf90c58a64e1d9f368b728436fc6d5c))
* Add RandomChoice augmentation. ([92848d8](https://github.com/AmbiqAI/helia-edge/commit/92848d885495ebafdfa9870bf4890036009cb65b))
* Add release GHA. ([a497cc1](https://github.com/AmbiqAI/helia-edge/commit/a497cc1a79a8f7355f16b1dfb18e39daf2065ba3))
* Add simply multi-f1score wrapper. ([d1a26fa](https://github.com/AmbiqAI/helia-edge/commit/d1a26fae1217e1308e89cd0a41a2615d47a4426c))
* Add SNR metric. ([4f15a9f](https://github.com/AmbiqAI/helia-edge/commit/4f15a9f1413b089f3ee7f6538d68dd976a88a670))
* Add thresholding. ([59dc664](https://github.com/AmbiqAI/helia-edge/commit/59dc66410ff707ec1c929ebf6388e6c95567081d))
* Add utility functions. ([df9fc6f](https://github.com/AmbiqAI/helia-edge/commit/df9fc6ff5f82f5cf2ac533063bbcb68b229fdd4f))
* Add utility functions. ([a842eb9](https://github.com/AmbiqAI/helia-edge/commit/a842eb99183439fdffa7f3f4465dbdefe128e625))
* Adds TfLiteKerasInterpreter. ([9ea7d64](https://github.com/AmbiqAI/helia-edge/commit/9ea7d647287c11acd3333c5ac1071612e71ca145))
* Allow applying forward_backward. ([d1ff173](https://github.com/AmbiqAI/helia-edge/commit/d1ff173808cfa63797f6c5d65cba17c415f010b8))
* Allow augmentations to happen outside loop. ([e62d22f](https://github.com/AmbiqAI/helia-edge/commit/e62d22f649d3cbdad5ce581cac84c879290a6b4e))
* Allow augmentations to happen outside loop. ([047a53c](https://github.com/AmbiqAI/helia-edge/commit/047a53c156d7985be6a7c6b30272d94159e3644e))
* Allow augmentations to happen outside loop. ([8fa63ce](https://github.com/AmbiqAI/helia-edge/commit/8fa63ce446cc278ff58d2c69fbd57b7ec6dd83ec))
* CI/CD pipelines with docs. ([b708b1f](https://github.com/AmbiqAI/helia-edge/commit/b708b1f6c05727ba7489846fb70182b5d2fc6ebc))
* Helper function to append layers to existing model. ([13e0a7a](https://github.com/AmbiqAI/helia-edge/commit/13e0a7a213cabe383af00146e41f396572002e89))
* History plot ([14e80b9](https://github.com/AmbiqAI/helia-edge/commit/14e80b9b3ad30418492b64d97be46b102743d1f9))
* Lots of 1D preprocessing/augmentation layers. ([f6759ab](https://github.com/AmbiqAI/helia-edge/commit/f6759abad43d45675a0b3dd57e7970ca3db5f01e))
* Make base augmentations that can be chained together. Consolidate preprocessing to 1D and 2D. ([a3c339c](https://github.com/AmbiqAI/helia-edge/commit/a3c339c8e00c0e45a5bc4443816321784ff95809))
* Make base augmentations that can be chained together. Consolidate preprocessing to 1D and 2D. ([0feca1f](https://github.com/AmbiqAI/helia-edge/commit/0feca1f6337dd2e4a4811332a3e9945f2f4336b7))
* Migrate initial models. ([839172f](https://github.com/AmbiqAI/helia-edge/commit/839172f282d415ebf24ddd1e52abc51a4417a8bd))
* Migrate model doc info to KerasEdge. ([8392991](https://github.com/AmbiqAI/helia-edge/commit/8392991b7efcb5774b56dc1e57f8916ccf904de9))
* Migrate poetry to uv. ([804ad04](https://github.com/AmbiqAI/helia-edge/commit/804ad0456ae0869431cb166801abdf69d11fa637))
* Migrate poetry to uv. ([259d42e](https://github.com/AmbiqAI/helia-edge/commit/259d42e647f25ddf9795b76cd1bb8000858642d0))
* More utils. ([398ad2c](https://github.com/AmbiqAI/helia-edge/commit/398ad2c7bbe73a6e70504f847d663fba0dd5136e))
* Renamed package. ([#2](https://github.com/AmbiqAI/helia-edge/issues/2)) ([4dfa67a](https://github.com/AmbiqAI/helia-edge/commit/4dfa67a5d17190d803d8ef744c72a4c5dc04e0b6))
* Setup poetry. ([a89e3c7](https://github.com/AmbiqAI/helia-edge/commit/a89e3c75b55e464572ac7e8ed4de3adfabaefbb0))
* Simple id generator routines. ([9026dbf](https://github.com/AmbiqAI/helia-edge/commit/9026dbf484ec04709294a90cb9458070c46ceede))
* Update docs. ([f3e617b](https://github.com/AmbiqAI/helia-edge/commit/f3e617bcbffc81bef28ffe69835778c3bfc3a08c))
* Update lock file. ([564e550](https://github.com/AmbiqAI/helia-edge/commit/564e55006d724dc9b1168d0e7abeed8f38b223fa))
* Update readme. ([69e68ca](https://github.com/AmbiqAI/helia-edge/commit/69e68ca0292885c41e950d60bd159101ad22afae))
* Use fixed loop limit for now. ([82690a3](https://github.com/AmbiqAI/helia-edge/commit/82690a3e53538c5123cd4c35e23f5eec7855db5f))
* Use uniform distribution for amplitude. ([b27f74f](https://github.com/AmbiqAI/helia-edge/commit/b27f74f1ff6d208666ec8ca418e6bbe187860bb8))


### Bug Fixes

* Coerce dtype. ([8d4bbad](https://github.com/AmbiqAI/helia-edge/commit/8d4bbad84b0e2e413cee5b456d79c647794d352e))
* Correct augmentation bugs. ([7bf444d](https://github.com/AmbiqAI/helia-edge/commit/7bf444de1af77eadf46b87eadb2115ca6899edfd))
* Correct model reshape. ([8ed08fc](https://github.com/AmbiqAI/helia-edge/commit/8ed08fca7f34e2c69ad7d8f65f8db34eb06db8ab))
* Correct output dtype for tflite interpreter. ([2615999](https://github.com/AmbiqAI/helia-edge/commit/2615999e070c3ce4819dec309ba16217268699f8))
* Correct RP config and workflow. ([da4ac06](https://github.com/AmbiqAI/helia-edge/commit/da4ac06a12bbf72e9b27cc42bc8f1ea7709c3946))
* Correct tflite conversion. ([2eb733d](https://github.com/AmbiqAI/helia-edge/commit/2eb733d5472ef946fc3c7733c2ba27af9d340b6d))
* Correct warping ([ebe21a9](https://github.com/AmbiqAI/helia-edge/commit/ebe21a9a729c503e2740d5343ab1e93b913da1b8))
* Ensure sine wave shape matches input. ([5ddcf0b](https://github.com/AmbiqAI/helia-edge/commit/5ddcf0b97f2ab82a124ca3853e98047b19560ebf))
* Ensure unique layer names. ([51dcab8](https://github.com/AmbiqAI/helia-edge/commit/51dcab8489a9aa6494d17be2d31ae56649da78c5))
* handle odd differences between train_step and test_step. ([64065b7](https://github.com/AmbiqAI/helia-edge/commit/64065b75403dc4e33f505bc9c63898362ba8a888))
* handle odd differences between train_step and test_step. ([9d88a89](https://github.com/AmbiqAI/helia-edge/commit/9d88a8965881df82d7abb15a20c1a35ab9abd7e7))
* Improve model definitions. ([197f515](https://github.com/AmbiqAI/helia-edge/commit/197f515f5ff8845813cab36caf7f348b80d6d15f))
* Only CI dependencies ([6fe2600](https://github.com/AmbiqAI/helia-edge/commit/6fe26004467bdd732e31e5dd516026bbfc34a3fa))
* Only CI dependencies ([bbceb2c](https://github.com/AmbiqAI/helia-edge/commit/bbceb2cada874c0bb127c21ec0bc768a00d40d1a))
* Reduce logging outside of module. ([62740ac](https://github.com/AmbiqAI/helia-edge/commit/62740acd4fc47338afb3bdd476309359b3bf25ab))
* Update test step. ([335c1f5](https://github.com/AmbiqAI/helia-edge/commit/335c1f54c8fea08ca13af595d25247c7c19b00d1))
* Update test step. ([b9c58a9](https://github.com/AmbiqAI/helia-edge/commit/b9c58a97700be1192d56f555ea516f59a782be33))
