# Home

<!-- <p align="center">
  <a href="https://github.com/AmbiqAI/helia-edge"><img src="./assets/app-banner.png" alt="HeliaEdge"></a>
</p> -->

[![](./assets/app-banner-light.png#only-dark)](https://ambiqai.github.io/helia-edge/)
[![](./assets/app-banner-dark.png#only-light)](https://ambiqai.github.io/helia-edge/)

---

**Documentation**: <a href="https://ambiqai.github.io/helia-edge" target="_blank">https://ambiqai.github.io/helia-edge</a>

**Source Code**: <a href="https://github.com/AmbiqAI/helia-edge" target="_blank">https://github.com/AmbiqAI/helia-edge</a>

---


<img style="float: left; margin: 0px 15px 15px 0px;" src="./assets/app-logo.png" width="96px" />

**HeliaEdge** is a [Keras 3](https://keras.io) add-on focused on training and deploying models on resource-constrained, edge devices. HeliaEdge relies heavily on [Keras 3](https://keras.io) leveraging its multi-backend support and customizable architecture. This package provides a variety of additional models, layers, optimizers, quantizers, and other components to help users train and deploy models for edge devices.

<br style="clear:both" />

<div class="grid cards" markdown>

- :material-rocket-launch: [Getting Started](usage/index.md)
- :material-api: [API Documentation](api/helia_edge)
- :simple-docsdotrs: [Usage Examples](examples/index.md)
- :simple-jupyter: [Explore Guides](guides/index.md)

</div>

## Main Features

* [**Callbacks**](api/helia_edge/callbacks): Training callbacks
* [**Converters**](api/helia_edge/converters): Converters for exporting models
* [**Interpreters**](api/helia_edge/interpreters): Inference engine interpreters (e.g. TFLite)
* [**Layers**](api/helia_edge/layers): Custom layers including `tf.data.Dataset` preprocessing layers
* [**Losses**](api/helia_edge/losses): Additional losses such as SimCLRLoss
* [**Metrics**](api/helia_edge/metrics): Custom metrics such as SNR
* [**Models**](api/helia_edge/models): Highly parameterized 1D/2D model architectures
* [**Optimizers**](api/helia_edge/optimizers): Additional optimizers
* [**Plotting**](api/helia_edge/plotting): Plotting routines
* [**Quantizers**](api/helia_edge/quantizers): Quantization techniques
* [**Trainers**](api/helia_edge/trainers): Custom trainers such as SSL contrastive learning
* [**Utils**](api/helia_edge/utils): Utility functions

## Problems **HeliaEdge** looks to solve

### Compatability issues between frameworks and inference engines

- [x] By leveraging Keras 3, entire workflows can be run using a variety of backends using a consistent front-end API. This allows selecting a backend that plays nicely with a specific inference engine without rewriting the entire model.

### SOTA models dont scale down well and come in limited configurations

- [x] By providing highly parameterized model architectures based on SOTA models, users can easily scale down models to fit their needs.

### Limited 1D time-series models

- [x] Most included models in HeliaEdge provide both 1D and 2D versions. The package also contains time-series specific models.

### Limited support for quantization, pruning, and other model optimization techniques

- [x] HeliaEdge provides a variety of quantization and pruning techniques to optimize models for edge deployment.
