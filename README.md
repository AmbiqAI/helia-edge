<p align="center">
  <a href="https://github.com/AmbiqAI/helia-edge"><img src="./docs/assets/app-banner-dark.png" alt="HeliaEdge"></a>
</p>

**Documentation**: <a href="https://ambiqai.github.io/helia-edge" target="_blank">https://ambiqai.github.io/helia-edge</a>

**Source Code**: <a href="https://github.com/AmbiqAI/helia-edge" target="_blank">https://github.com/AmbiqAI/helia-edge</a>

---

**HeliaEdge** is a [Keras 3](https://keras.io) add-on focused on training and deploying models on resource-constrained, edge devices. Helia relies heavily on [Keras 3](https://keras.io) leveraging its multi-backend support and customizable architecture. This package provides a variety of additional models, layers, optimizers, quantizers, and other components to help users train and deploy models for edge devices.



- **[Getting Started](usage/index.md)**: Learn how to install and use HeliaEdge
- **[API Documentation](api/helia_edge)**: Explore the API
- **[Usage Examples](examples/index.md)**: See examples of HeliaEdge in action
- **[Explore Guides](guides/index.md)**: View in-depth guides on using HeliaEdge

## Main Features

* [**Callbacks**](https://ambiqai.github.io/helia-edge/api/helia_edge/callbacks): Training callbacks
* [**Converters**](https://ambiqai.github.io/helia-edge/api/helia_edge/converters): Converters for exporting models
* [**Interpreters**](https://ambiqai.github.io/helia-edge/api/helia_edge/interpreters): Inference engine interpreters (e.g. TFLite)
* [**Layers**](https://ambiqai.github.io/helia-edge/api/helia_edge/layers): Custom layers including `tf.data.Dataset` preprocessing layers
* [**Losses**](https://ambiqai.github.io/helia-edge/api/helia_edge/losses): Additional losses such as SimCLRLoss
* [**Metrics**](https://ambiqai.github.io/helia-edge/api/helia_edge/metrics): Custom metrics such as SNR
* [**Models**](https://ambiqai.github.io/helia-edge/api/helia_edge/models): Highly parameterized 1D/2D model architectures
* [**Optimizers**](https://ambiqai.github.io/helia-edge/api/helia_edge/optimizers): Additional optimizers
* [**Plotting**](https://ambiqai.github.io/helia-edge/api/helia_edge/plotting): Plotting routines
* [**Quantizers**](https://ambiqai.github.io/helia-edge/api/helia_edge/quantizers): Quantization techniques
* [**Trainers**](https://ambiqai.github.io/helia-edge/api/helia_edge/trainers): Custom trainers such as SSL contrastive learning
* [**Utils**](https://ambiqai.github.io/helia-edge/api/helia_edge/utils): Utility functions

## Problems **HeliaEdge** looks to solve

### Compatability issues between frameworks and inference engines

- [x] By leveraging Keras 3, entire workflows can be run using a variety of backends using a consistent front-end API. This allows selecting a backend that plays nicely with a specific inference engine without rewriting the entire model.

### SOTA models dont scale down well and come in limited configurations

- [x] By providing highly parameterized model architectures based on SOTA models, users can easily scale down models to fit their needs.

### Limited 1D time-series models

- [x] Most included models in HeliaEdge provide both 1D and 2D versions. The package also contains time-series specific models.

### Limited support for quantization, pruning, and other model optimization techniques

- [x] HeliaEdge provides a variety of quantization and pruning techniques to optimize models for edge deployment.
