# Pruning Aware Training

Pruning Aware Training is an easy-to-integrate framework for optimizing internal networks. It accelerates inference with minimal task degradation by removing redundant parameters. Currently, it supports:
- **Structured (Channel) Pruning:** Removing entire channels (filters) for universal acceleration.

## Project Objectives

- **Reduce Model Size:** Prune redundant parameters while preserving accuracy.
- **Accelerate Inference:** Lower computational overhead for faster deployment.

## Project Structure

- **torch-pruning:**  
  Contains the original torch-pruning code along with our enhancements for channel pruning.

- **pruning_utils.py (under torch-prunig):**  
  Core module orchestrating the pruning process (initialization, regularization, and epoch-wise pruning).

- **Documentation:**  
  In-depth explanations, usage examples, and detailed descriptions of supported methods.  
  - See [Documentation/README.md](Documentation/README.md) for complete guidance.

## Installation & Usage

### Installation

1. ** Classic pip usage**
   ```
   pip install [LINK]
   ```

### Minimal Code Integration

1. **Initialization:**  
   Instantiate the Pruner with your model:
   ```python
   from torch_pruning.pruning_utils import Pruning
   pruner = Pruning(model, output_dir, device=device)
   ```

2. **Regularization:**  
   After the backward pass, add pruning regularization:
   ```python
   loss.backward()
   pruner.channel_regularize(model)
   ```

3. **Pruning Step:**  
   Call the prune method at each epoch (or step):
   ```python
   pruner.prune(model, epoch)
   ```

## Updates & Support