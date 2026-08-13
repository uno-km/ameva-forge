# AMEVA-Forge

> **Release 1 (Public Technical Preview)**: Browser-local Educational Autograd & Small Model Experimentation Framework

![AMEVA Forge Logo](https://img.shields.io/badge/WebGPU-Powered-blueviolet?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python) ![Pyodide](https://img.shields.io/badge/Browser_Ready-Pyodide-yellow?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

AMEVA-Forge is a lightweight, PyTorch-like deep learning framework designed for **educational purposes and small model experimentation** directly inside the browser using Pyodide and WebGPU.

## 🚀 Release 1 Scope: Technical Preview

AMEVA-Forge is currently in an **Internal Alpha / Technical Preview** stage. Our Release 1 goal is laser-focused on one mission: 
**"Enable students and researchers to build, train, and understand a 2-layer Multi-Layer Perceptron (MLP) entirely in the browser using Python."**

Unlike heavy production frameworks, AMEVA-Forge requires zero C++ infrastructure or server backends. It provides a pure Python interface that dispatches to WebGPU.

### Supported Features (Release 1)
- **Core Tensor Ops**: `tensor`, `upload`, `readback`, `dispose`
- **Basic Math**: `add`, `sub`, `mul`, `div`, `neg`
- **Matrix Operations**: `matmul` (2D)
- **Activations**: `relu`
- **Shape & Reductions**: `reshape`, `transpose` (2D), scalar `sum` *(Note: under active stability fixes)*
- **Training**: Full reverse-mode Autograd and SGD Optimizer

*(Note: Convolutional layers, RNNs, and complex pooling are intentionally excluded from Release 1 to ensure mathematical correctness and stability of the core MLP engine.)*

## 💻 Quick Start

### Installation
```bash
pip install ameva-forge
```
*(Requires a Pyodide/WebGPU compatible browser environment for execution)*

### 2-Layer MLP Example
```python
import forge as fg

# 1. Initialize tensors (Requires WebGPU Browser Env)
x = fg.randn((2, 3), requires_grad=True)
w = fg.randn((3, 4), requires_grad=True)
b = fg.zeros((2, 4), requires_grad=True)

# 2. Forward pass
out = x @ w + b
loss = fg.nn.MSELoss()(out, fg.ones_like(out))

# 3. Backward pass (Autograd)
loss.backward()

# 4. SGD Step
lr = 0.01
w.data -= lr * w.grad.data
b.data -= lr * b.grad.data
```

## 🤝 Roadmap
- **Phase 1 (Current)**: Stabilize core MLP mathematical parity (CPU/GPU) and WebGPU resource limits.
- **Phase 2**: Introduce comprehensive Headless WebGPU E2E CI testing.
- **Phase 3**: Extend support for experimental Vision models and advanced optimizations.

---

### 🔍 Search Keywords
`WebGPU Deep Learning`, `Browser-based AI Training`, `Pyodide Machine Learning`, `Educational Autograd Framework`, `Python WebGPU`, `Edge AI Training`, `Browser PyTorch Alternative`, `Wasm Python Deep Learning`
