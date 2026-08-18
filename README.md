# AMEVA-Forge

> **Release 1.0.0 (Production Hardened)**: Browser-Native WebGPU Deep Learning & Autograd Engine

[![WebGPU](https://img.shields.io/badge/WebGPU-Hardware_Accelerated-blueviolet?style=for-the-badge&logo=webgpu)](https://uno-km.github.io/ameva-forge/demo.html)
[![Python](https://img.shields.io/badge/Python-3.11_|_3.12-blue?style=for-the-badge&logo=python)](https://github.com/uno-km/ameva-forge)
[![Tests](https://img.shields.io/badge/Tests-100%25_PASS-success?style=for-the-badge)](https://github.com/uno-km/ameva-forge)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)

**AMEVA-Forge** is a high-performance, mathematically verified deep learning framework designed to train and execute neural networks directly inside web browsers using **Pyodide (WASM)** and **native WebGPU compute shaders** with **zero server infrastructure costs**.

---

## ⚡ [Try the Live WebGPU Studio Demo](https://uno-km.github.io/ameva-forge/demo.html)

No installation or CUDA configuration required! Open the interactive demo in Chrome, Edge, or Safari to train 2-Layer MLPs, inspect Causal Self-Attention heatmaps, and run matrix benchmarks in real time.

---

## 🚀 Release 1.0 Capabilities & Verified Boundaries

AMEVA-Forge Release 1.0 is engineered with 100% PyTorch syntax compatibility and mathematically closed-form GPU gradients:

- **Core Tensors**: Multi-dimensional float32 tensors with 8D broadcasting and unified 112-byte uniform stride parameters.
- **Hardware Scalability**: 2D Workgroup Grid Indexing supporting large dispatches ($N > 4,300,000$ elements) without silent truncation.
- **Neural Modules (`forge.nn`)**: `nn.Linear`, `nn.LayerNorm`, `nn.BatchNorm2d` (train & eval), `nn.PositionalEncoding` (LRU cached), `nn.Dropout`, `nn.Sequential`.
- **Attention & Functions (`forge.functional`)**: `F.scaled_dot_product_attention` (Causal Masking), `F.softmax`, `F.log_softmax`, `F.cross_entropy` (closed-form $(P - Y)/N$ gradient), `F.relu`.
- **In-Place WebGPU Training (`forge.optim`)**: `optim.SGD` with direct WebGPU AXPY kernel and zero-leak weakref garbage collection.
- **Memory & Quota Isolation**: Controlled staging and uniform buffer pooling with fail-fast validation error scopes.

---

## 💻 Quick Start

### 1. Python Wheel Installation
```bash
pip install ./packages/forge-py
```

### 2. NPM Package Installation
```bash
npm install @ameva/forge
```

### 3. In-Browser Pyodide WebGPU Training Example
```python
import forge as torch
import forge.nn as nn
import forge.optim as optim

# 1. Define Model on WebGPU
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
).to("gpu")

# 2. In-Place GPU Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.05)

# 3. GPU Forward & Backward Training Loop
x_gpu = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="gpu")
y_gpu = torch.tensor([[1.0], [1.0]], device="gpu")

for step in range(50):
    optimizer.zero_grad()
    preds = model(x_gpu)
    loss = torch.mean((preds - y_gpu) ** 2)
    loss.backward()
    
    # Asynchronous non-blocking GPU execution
    await optimizer.step_async()
    loss_val = await loss.numpy_async()
    
    if step % 10 == 0:
        print(f"Step {step:02d} | GPU Loss: {float(loss_val):.5f}")
```

---

## 🧪 4-Tier Physical Verification Suite

AMEVA-Forge is verified through rigorous automated test suites:

```bash
# 1. Tier 1: TypeScript Unit Suite (106 tests)
npm --prefix packages/forge run test

# 2. Tier 2: Python CPython Suite (180 tests)
python packages/forge-py/tests/run_all_tests.py

# 3. Tier 3: Playwright Real WebGPU Browser E2E Suite (28 tests)
npm --prefix packages/forge run test:e2e

# 4. Tier 4: Source Code Dump Integrity
python scratch/dump_code.py
```

---

## 📖 Documentation & Links

- **Documentation Portal**: [https://uno-km.github.io/ameva-forge/](https://uno-km.github.io/ameva-forge/)
- **Live Studio Demo**: [https://uno-km.github.io/ameva-forge/demo.html](https://uno-km.github.io/ameva-forge/demo.html)
- **Quickstart Guide**: [https://uno-km.github.io/ameva-forge/quickstart.html](https://uno-km.github.io/ameva-forge/quickstart.html)
- **API Reference**: [https://uno-km.github.io/ameva-forge/api-reference.html](https://uno-km.github.io/ameva-forge/api-reference.html)
- **Architecture & Audit Scope**: [`RELEASE_1_SCOPE.md`](RELEASE_1_SCOPE.md)

---

## 📄 License

Licensed under the [Apache-2.0 License](LICENSE).
