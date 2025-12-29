![CI](https://github.com/AvrahamRaviv/PruningAwareTraining/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/AvrahamRaviv/PruningAwareTraining?branch=main)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

# PruningAwareTraining

PruningAwareTraining is a research-oriented Python library for **pruning-aware training of deep neural networks**.
It focuses on integrating **structured pruning decisions directly into the training process**, rather than applying
pruning only as a post-training step.

The library is designed for researchers and practitioners working on model compression, structured pruning,
hardware-aware optimization, and efficient deep learning, with a particular emphasis on computer vision models.

---

## Background and Motivation

Modern deep neural networks typically contain **far more parameters than required** for effective representation.
While over-parameterization helps optimization and generalization, it also results in excessive computational
and memory costs at inference time.

Training small networks from scratch is often **harder to optimize and slower to converge** than starting from
larger models and gradually reducing their capacity.
This observation motivates **pruning during training**, where sparsity is introduced progressively while
preserving favorable optimization dynamics.

Model compression methods generally fall into three categories:
- Neural Architecture Search (NAS)
- Quantization
- Pruning

Pruning is particularly attractive because it allows models to retain their original architecture while
removing redundant parameters.

---

## Pruning Paradigms and Their Limitations

### Unstructured and Pattern Pruning

Work such as the Lottery Ticket Hypothesis (LTH) has shown that up to 90% of weights can be removed without
significant accuracy degradation.
However, most unstructured and pattern-based pruning methods rely on **zeroing weights**, leaving the
computational graph unchanged.

As a result, these methods provide **little to no real runtime or hardware benefit** unless supported by
specialized sparse hardware.

PyTorch’s native pruning utilities (`torch.nn.utils.prune`) exemplify this approach:
- Pruned weights are masked, not removed
- The execution graph remains intact
- Useful for sparse-model research and fine-tuning
- Limited practical inference gains on standard hardware

---

## Structured Pruning and Torch-Pruning

Structured pruning removes entire channels, filters, or layers, making it **naturally compatible with hardware
and runtime acceleration**.
This approach, however, introduces strong structural constraints and requires careful dependency management.

The Torch-Pruning library  
https://github.com/VainF/Torch-Pruning  
is a strong and mature foundation in this space. It:
- Physically modifies the model graph to remove pruned structures
- Uses a dependency graph to ensure correctness
- Implements multiple state-of-the-art pruning algorithms
- Supports real, executable pruned models (not just masking)

At the same time, Torch-Pruning primarily targets **post-training pruning workflows**.
Iterative pruning **during training**, integration with training loops, and production-oriented abstractions
are intentionally left to the user.

---

## Statement of Need

PruningAwareTraining builds on top of Torch-Pruning and addresses a missing layer in the ecosystem:
**a generic, flexible, and production-aware framework for pruning during training**.

The library combines the strengths of mask-based pruning (ease of integration, training-time flexibility)
with the strengths of structured pruning (real graph modification and hardware gains), by introducing
an additional abstraction layer that manages pruning-aware optimization.

The goal is not to replace Torch-Pruning, but to **extend it into the training regime**, enabling:
- Iterative pruning across epochs
- Seamless integration with optimizers and regularization
- Reproducible pruning-aware training pipelines
- Clean save/load of pruned models

---

## Design Philosophy

- **Single, generic API** for pruning-aware training
- Built on top of Torch-Pruning’s dependency graph and real pruning
- Installed via standard `pip`
- Easy integration using configuration files
- Supports multiple pruning strategies and schedules
- Designed to be research-friendly but production-conscious

---

## Pruning-Aware Training Workflow

A typical workflow consists of three stages:

1. **Regularization**  
   Encourage structured sparsity during training.

2. **Iterative Pruning / Masking**  
   Gradually prune structures based on the training signal and constraints.

3. **Fine-Tuning**  
   Stabilize and recover accuracy after pruning steps.

This workflow enables pruning decisions to interact directly with optimization,
rather than being applied as an isolated post-processing step.

---

## Features

- Iterative structured pruning during training
- Integration with Torch-Pruning dependency graphs
- Support for saving and loading pruned models
- Production-friendly pruned execution graphs
- Compatibility with quantization-aware training (QAT)
- ONNX export of pruned models
- Configuration-driven experimentation

---

## Installation

```bash
git clone https://github.com/AvrahamRaviv/PruningAwareTraining.git
cd PruningAwareTraining
pip install -e .
```

---

## 🚀 Quick Start

```python
from pruningawaretraining import Pruner
from model import model, optimizer

pruner = Pruner(
    model,
    target_sparsity=0.5,
    schedule="linear"
)

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer)
    pruner.step()
```
This example illustrates how pruning decisions are integrated directly into the training loop.

---

## 🧪 Example Config (JSON)

```json
{
  "start_epoch": 5,
  "end_epoch": 50,
  "interval": 5,
  "global_sparsity": 0.5,
  "mac_target": 0.7,
  "layers_to_prune": ["conv1", "layer2", "layer3"]
}
```

---
## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Please open a pull request or report issues via the [GitHub Issues](../../issues) page.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🧭 Acknowledgements

Built upon [Torch-Pruning](https://github.com/VainF/Torch-Pruning) and inspired by its DepGraph architecture.
Special thanks to collaborators and the research community for shaping this work.

---

