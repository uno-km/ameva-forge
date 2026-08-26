# AMEVA-Forge (`ameva-forge`)

<div align="center">

[![Official Documentation](https://img.shields.io/badge/docs-uno--km.vercel.app%2Flib%2Fforge-004499?style=for-the-badge&logo=vercel)](https://uno-km.vercel.app/lib/forge/)
[![PyPI version](https://img.shields.io/pypi/v/ameva-forge.svg?style=for-the-badge&color=007ec6&logo=pypi&logoColor=white)](https://pypi.org/project/ameva-forge/)
[![Python](https://img.shields.io/badge/Python-3.9_|_3.10_|_3.11_|_3.12_|_3.13-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/ameva-forge/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Pure_WGSL_Compute-blueviolet?style=for-the-badge&logo=webgpu)](https://uno-km.vercel.app/lib/forge/)
[![Tests](https://img.shields.io/badge/Tests-292%2F292_Passed_(100%25)-brightgreen?style=for-the-badge)](https://uno-km.vercel.app/lib/forge/benchmarks.html)
[![Open Collective](https://img.shields.io/badge/Open_Collective-AOSF_Fund-004499?style=flat&logo=opencollective)](https://opencollective.com/ameva-fund)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub_Sponsors-uno--km-ea4aaa?style=flat&logo=githubsponsors)](https://github.com/sponsors/uno-km)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=for-the-badge)](LICENSE)
[![AMEVA Foundation](https://img.shields.io/badge/Foundation-AOSF_Tier_1-orange?style=for-the-badge)](https://uno-km.vercel.app/docs/foundation/)

### High-Performance Client-Side Deep Learning Engine & WebGPU Reverse-Mode Autograd Framework
**An Official Tier 1 Top-Level Open-Source Project of the AMEVA Foundation (AOSF)**

[🚀 Live WebGPU Studio Demo](https://uno-km.vercel.app/lib/forge/demo.html) • [📚 Official Documentation](https://uno-km.vercel.app/lib/forge/) • [📦 PyPI Package](https://pypi.org/project/ameva-forge/) • [💬 Issue Tracker](https://github.com/uno-km/ameva-forge/issues)

</div>

---

## ⚡ 1-Line Installation

```bash
pip install ameva
```

Or run directly inside any modern web browser via WebGPU & Pyodide (Zero Installation, Zero Server Cost):

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script src="https://uno-km.vercel.app/lib/forge/dist/forge-py-bundle.js"></script>
```

---

## 🏛️ Architectural Overview

AMEVA-Forge connects a deterministic Python autograd frontend directly to direct-to-silicon WebGPU WGSL compute shaders:

```text
+-----------------------------------------------------------------------------------+
|                            AMEVA-Forge User Space                                 |
|   forge.nn  |  forge.optim  |  forge.linalg  |  forge.fft  |  forge.distributions |
+-----------------------------------------------------------------------------------+
|                      Reverse-Mode Autograd DAG Engine                             |
|       Vector-Jacobian Products (VJP)  *  In-Place Mutation Version Locks          |
+-----------------------------------------------------------------------------------+
|                         Hardware Abstraction Layer                                |
|   CPU Backend (Vectorized C/NumPy)  <--->  WebGPU Backend (Async WGSL Kernels)    |
|   Staging Buffer Recycling Pool     <--->  Zero-Leak Allocation Token Ring        |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Key Capabilities & Verified Boundaries

1. **Deterministic Autograd & Topological Execution**  
   Strict reverse-mode automatic differentiation graph with cycle detection, multi-output tuple bindings, in-place version invalidation, and scalar-tensor memory optimization.

2. **WebGPU Hardware-Accelerated WGSL Compute Pipeline**  
   Custom WGSL compute shaders for fused matrix multiplication, tensor reduction, element-wise broadcasting, convolutions, LayerNorm, and Softmax operating directly on client GPU hardware.

3. **100% Client-Side In-Browser Deep Learning**  
   Train neural networks directly inside browser tabs with Pyodide WebAssembly and WebGPU without server GPUs, APIs, or cloud costs.

4. **Zero-Copy Buffer Recycling & Memory Token Pools**  
   Direct GPU buffer lifecycle management with zero memory leaks, reusable staging buffers, and asynchronous queue dispatching.

---

## 📦 PyTorch Compatibility API Example

```python
import ameva.forge as forge
import ameva.forge.nn as nn
import ameva.forge.optim as optim

# 1. Define Model
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# 2. Instantiate on WebGPU device
model = TinyNet().to("webgpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. Training step in browser
inputs = forge.randn(32, 784, device="webgpu", requires_grad=False)
targets = forge.randint(0, 10, (32,), device="webgpu")

optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print(f"WebGPU Step Complete! Loss: {loss.item():.4f}")
```

---

## 📄 License

Apache-2.0 / MIT License © 2026 AMEVA Open-Source Foundation (AOSF). All Rights Reserved.


---

## 💖 Sponsorship & Community Backing

AMEVA is an independent open-source public good governed under the **AMEVA Open-Source Foundation (AOSF)**. All sponsorship funds are 100% publicly audited and dedicated to physical ARM64 testbeds and CI/CD GPU runners.

- **Open Collective (Non-Profit 501(c)(6))**: [https://opencollective.com/ameva-fund](https://opencollective.com/ameva-fund)
- **GitHub Sponsors**: [https://github.com/sponsors/uno-km](https://github.com/sponsors/uno-km)
- **Official Foundation Portal**: [https://uno-km.vercel.app/docs/foundation/sponsorship.html](https://uno-km.vercel.app/docs/foundation/sponsorship.html)
