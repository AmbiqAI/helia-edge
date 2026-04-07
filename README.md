# heliaEDGE

> Build edge-ready ML systems on top of Keras 3, from training to deployment.

[![CI](https://github.com/AmbiqAI/helia-edge/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/AmbiqAI/helia-edge/actions/workflows/ci.yaml)
[![Docs](https://github.com/AmbiqAI/helia-edge/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/AmbiqAI/helia-edge/actions/workflows/docs.yaml)
[![Release](https://github.com/AmbiqAI/helia-edge/actions/workflows/release.yaml/badge.svg)](https://github.com/AmbiqAI/helia-edge/actions/workflows/release.yaml)

heliaEDGE is an open-source toolkit from Ambiq for developers shipping ML on resource-constrained devices. It extends Keras 3 with edge-focused building blocks so you can design, train, optimize, and deploy models without stitching together a fragile stack of custom scripts.

The goal is simple: keep the developer experience high while making edge deployment practical.

---

## Why use **heliaEDGE**?

- **Stay inside Keras 3**: Keep familiar model/training APIs while adding edge-specific capabilities.
- **Design for constraints early**: Build models with architecture choices that map better to edge memory, latency, and power limits.
- **Reduce integration overhead**: Use one toolkit for modeling, training utilities, optimization, conversion, and evaluation.
- **Move faster to deployment**: Spend less time on glue code and more time on model quality.

### Start here

- **[Getting Started](https://ambiqai.github.io/helia-edge/usage/)**: Install and run your first heliaEDGE workflow
- **[API Documentation](https://ambiqai.github.io/helia-edge/api/helia_edge/)**: Explore the full API surface
- **[Guides](https://ambiqai.github.io/helia-edge/guides/)**: Follow practical end-to-end walkthroughs

## Main Features

* [**Callbacks**](https://ambiqai.github.io/helia-edge/api/helia_edge/callbacks): Training lifecycle and monitoring helpers
* [**Converters**](https://ambiqai.github.io/helia-edge/api/helia_edge/converters): Export pipelines for deployment targets
* [**Interpreters**](https://ambiqai.github.io/helia-edge/api/helia_edge/interpreters): Runtime inference interfaces (including TFLite)
* [**Layers**](https://ambiqai.github.io/helia-edge/api/helia_edge/layers): Edge-centric layers, including data preprocessing components
* [**Losses**](https://ambiqai.github.io/helia-edge/api/helia_edge/losses): Additional losses for modern training workflows
* [**Metrics**](https://ambiqai.github.io/helia-edge/api/helia_edge/metrics): Extended evaluation metrics for edge model analysis
* [**Models**](https://ambiqai.github.io/helia-edge/api/helia_edge/models): Parameterized 1D/2D architectures for flexible scaling
* [**Optimizers**](https://ambiqai.github.io/helia-edge/api/helia_edge/optimizers): Optimization options beyond core Keras defaults
* [**Plotting**](https://ambiqai.github.io/helia-edge/api/helia_edge/plotting): Visualization helpers for training and evaluation outputs
* [**Quantizers**](https://ambiqai.github.io/helia-edge/api/helia_edge/quantizers): Quantization workflows for efficient inference
* [**Trainers**](https://ambiqai.github.io/helia-edge/api/helia_edge/trainers): Trainer abstractions including self-supervised patterns
* [**Utils**](https://ambiqai.github.io/helia-edge/api/helia_edge/utils): Practical utilities for common ML/edge tasks

## Built for Developers Shipping to Edge

heliaEDGE is a strong fit when you need to:

- Build models that are both accurate and deployable on constrained hardware.
- Support multiple model styles (classification, time-series, autoencoders, SSL) in one codebase.
- Keep your team on standard Keras workflows while still handling edge-specific requirements.
- Prototype quickly, then iterate with quantization and deployment constraints in mind.

### Typical Workflow

1. Choose or compose a model architecture suited for your task and constraints.
2. Train with heliaEDGE trainers, losses, callbacks, and metrics.
3. Evaluate quality and cost tradeoffs with plotting and custom metrics.
4. Apply quantization/optimization and export for your inference runtime.

### Next step

Visit the docs at **https://ambiqai.github.io/helia-edge/** and start with the **Getting Started** guide.
