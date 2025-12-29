---
title: "PruningAwareTraining: A Framework for Pruning-Aware Training with Structured Model Modification"
tags:
  - deep learning
  - model compression
  - structured pruning
  - pruning-aware training
  - computer vision
authors:
  - name: Avraham Raviv
    orcid: 0000-0002-4428-0505
    affiliation: 1
  - name: Yonatan Dinai
    orcid: 0009-0007-8994-054X
    affiliation: 1
  - name: Ishay Goldin
    orcid: 0009-0003-7883-032X
    affiliation: 1
  - name: Niv Zenghut
    orcid: 0009-0000-5623-8040
    affiliation: 1
affiliations:
  - name: Samsung Israel Research Center (SIRC)
    index: 1
date: 2025-12-29
bibliography: paper.bib
---

## Summary

PruningAwareTraining is a research-oriented Python library for **pruning-aware training of deep neural networks**.
The library enables structured pruning decisions to be integrated directly into the training process, rather than
applied only as a post-training operation.

The framework builds on top of Torch-Pruning, leveraging its dependency graph and real graph-modifying pruning
capabilities, while introducing higher-level abstractions for iterative pruning, regularization, and
training-time integration.
PruningAwareTraining targets research on efficient deep learning, structured pruning, and hardware-aware
optimization, with a focus on practical and reproducible experimentation.

---

## Statement of Need

Modern deep neural networks are typically over-parameterized, containing far more parameters than required for
effective representation.
While over-parameterization improves optimization and generalization, it leads to increased inference cost and
limits deployment on resource-constrained hardware.

Training compact models from scratch is often more difficult than starting from larger models and gradually
reducing their capacity.
This motivates **pruning during training**, where sparsity is introduced progressively while preserving favorable
optimization dynamics.

Existing pruning tools commonly fall into two categories.
Mask-based approaches, such as PyTorch’s native pruning utilities, zero out parameters while keeping the
computational graph intact, resulting in limited practical performance gains.
Structured pruning frameworks, such as Torch-Pruning, physically modify the execution graph and enable real
runtime benefits, but primarily target post-training workflows.

PruningAwareTraining addresses the gap between these approaches by providing a framework for **structured pruning
during training**.
It extends Torch-Pruning with abstractions for iterative pruning schedules, regularization-aware optimization,
and seamless integration into training loops, enabling pruning decisions to interact directly with learning
dynamics.

---

## Design and Functionality

PruningAwareTraining is designed as a lightweight abstraction layer on top of Torch-Pruning.
It relies on Torch-Pruning’s dependency graph to ensure correctness when removing channels or layers, while
exposing a higher-level API suitable for pruning-aware optimization.

Key design principles include:
- A single, generic API for pruning-aware training
- Explicit support for iterative pruning across epochs
- Integration with regularization-based sparsity induction
- Clean save and load of pruned models
- Compatibility with quantization-aware training and ONNX export

The framework is configuration-driven and designed to be easily integrated into existing training pipelines
with minimal code changes.

---

## Relation to Existing Work

Unstructured and pattern-based pruning methods, including those inspired by the Lottery Ticket Hypothesis,
have demonstrated that a large fraction of network parameters can be removed without degrading accuracy.
However, such methods typically rely on masking and do not alter the execution graph, limiting their practical
impact on standard hardware.

Structured pruning removes entire channels or layers and is naturally compatible with hardware acceleration,
but introduces strong structural constraints and requires careful dependency management.
Torch-Pruning provides a robust foundation for structured pruning by using dependency graphs to safely modify
model architectures.

PruningAwareTraining builds directly on Torch-Pruning, not as a replacement, but as an extension into the
training regime.
Its contribution lies in enabling structured pruning **during training**, rather than solely as a post-training
step, and in providing reusable abstractions for pruning-aware optimization workflows.

---

## Acknowledgements

The author thanks collaborators and colleagues for discussions on structured pruning, pruning-aware optimization,
and efficient deep learning.
