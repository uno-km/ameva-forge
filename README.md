# AMEVA-Forge

> **Release 1 Candidate (Internal Alpha)**: Browser-local Educational Autograd & Small Model Experimentation Framework

![AMEVA Forge Logo](https://img.shields.io/badge/WebGPU-Powered-blueviolet?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python) ![Pyodide](https://img.shields.io/badge/Browser_Ready-Pyodide-yellow?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

AMEVA-Forge is a lightweight, PyTorch-like deep learning framework designed for **educational purposes and small model experimentation** directly inside the browser using Pyodide and WebGPU.

## 🚀 Release 1 Scope: Technical Preview

AMEVA-Forge is currently in an **Internal Alpha / Technical Preview** stage. Our Release 1 goal is laser-focused on one mission: 
**"Enable students and researchers to build, train, and understand a 2-layer Multi-Layer Perceptron (MLP) in supported browser environments using Python."**

Unlike heavy production frameworks, AMEVA-Forge requires zero C++ infrastructure or server backends. It provides a pure Python interface that dispatches to WebGPU.

### Target Core Features for Release 1
- **Core Tensor Ops**: `tensor`, `upload`, `readback`, `dispose`
- **Basic Math**: `add`, `sub`, `mul`, `div`, `neg`
- **Matrix Operations**: `matmul` (2D)
- **Activations**: `relu`
- **Shape & Reductions**: `reshape`, `transpose` (2D), scalar `sum`
- **Training**: Full reverse-mode Autograd and SGD Optimizer

*(Note: Convolutional layers, RNNs, and complex pooling are intentionally excluded from Release 1 to ensure mathematical correctness and stability of the core MLP engine.)*

## 💻 Quick Start

### Installation
```bash
pip install ./packages/forge-py
```

### 1. Synchronous CPU Execution Example
```python
import forge as fg

x = fg.randn((2, 3), requires_grad=True)
w = fg.randn((3, 4), requires_grad=True)
b = fg.zeros((2, 4), requires_grad=True)

out = x @ w + b
loss = fg.nn.MSELoss()(out, fg.ones_like(out))
loss.backward()

optimizer = fg.optim.SGD([w, b], lr=0.01)
optimizer.step()
optimizer.zero_grad()
```

### 2. Async Browser WebGPU Execution Example
```python
import forge as fg
import forge.nn as nn
import forge.optim as optim

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# Move model parameters to GPU
model = model.to("gpu")

x_gpu = fg.tensor([[0.0, 1.0], [1.0, 0.0]]).to("gpu")
y_gpu = fg.tensor([[1.0], [1.0]]).to("gpu")

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

optimizer.zero_grad()
out = model(x_gpu)
loss = criterion(out, y_gpu)
loss.backward()

# GPU training requires async step
await optimizer.step_async()
loss_val = await loss.numpy_async()
print("WebGPU Loss:", loss_val.mean())
```

## 🤝 Roadmap
- **Phase 1 (Current)**: Stabilize core MLP mathematical parity (CPU/GPU) and WebGPU resource limits.
- **Phase 2**: Introduce comprehensive Headless WebGPU E2E CI testing.
- **Phase 3**: Extend support for experimental Vision models and advanced optimizations.

---

### 🔍 Search Keywords
`WebGPU Deep Learning`, `Browser-based AI Training`, `Pyodide Machine Learning`, `Educational Autograd Framework`, `Python WebGPU`, `Edge AI Training`, `Browser PyTorch Alternative`, `Wasm Python Deep Learning`
