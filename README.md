# AMEVA-Forge (`ameva-forge`)

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/ameva-forge.svg?style=for-the-badge&color=007ec6&logo=pypi&logoColor=white)](https://pypi.org/project/ameva-forge/)
[![PyPI Release](https://img.shields.io/badge/PyPI-v0.1.0_Live-28a745?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/ameva-forge/)
[![Python](https://img.shields.io/badge/Python-3.9_|_3.10_|_3.11_|_3.12_|_3.13-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/ameva-forge/)
[![WebGPU](https://img.shields.io/badge/WebGPU-Hardware_Accelerated-blueviolet?style=for-the-badge&logo=webgpu)](https://uno-km.github.io/ameva-forge/)
[![Tests](https://img.shields.io/badge/Tests-292%2F292_Passed_(100%25)-brightgreen?style=for-the-badge)](https://github.com/uno-km/ameva-forge)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![AMEVA Foundation](https://img.shields.io/badge/Foundation-AMEVA-orange?style=for-the-badge)](https://uno-km.github.io/ameva-forge/)

### High-Performance Client-Side Deep Learning Engine & WebGPU Reverse-Mode Autograd Framework
**An Official Open-Source Initiative of the AMEVA Foundation (아메바 재단)**

[🚀 Live WebGPU Studio Demo](https://uno-km.github.io/ameva-forge/demo.html) • [📚 Official Documentation](https://uno-km.github.io/ameva-forge/) • [📦 PyPI Package](https://pypi.org/project/ameva-forge/) • [💬 Issue Tracker](https://github.com/uno-km/ameva-forge/issues)

</div>

---

## ⚡ 1-Line Installation

```bash
pip install ameva
```

Or run directly inside any modern web browser via WebGPU & Pyodide (Zero Installation, Zero Server Cost):

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script src="https://uno-km.github.io/ameva-forge/dist/forge-py-bundle.js"></script>
```

---

## 🏛️ Architectural Overview

AMEVA-Forge connects a deterministic Python autograd frontend directly to direct-to-silicon WebGPU WGSL compute shaders:

```
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
2. **WebGPU Hardware Acceleration**  
   Direct-to-silicon WGSL compute shaders featuring 8-dimensional non-contiguous stride dispatching, 2D workgroup partitioning ($65,535 \times 65,535$), and explicit buffer lifecycle tracking.
3. **PyTorch 1:1 API Parity**  
   Seamless drop-in compatibility across neural layers (`nn.Module`, `nn.MultiheadAttention`, `nn.Conv2d`), mathematical primitives (`linalg`, `fft`, `special`), and probabilistic graphical models (`distributions`).
4. **Zero-Server Infrastructure (Edge & Browser)**  
   Execute full model fine-tuning and inference directly inside the browser using Pyodide and WebGPU with zero cloud compute cost and total data privacy.

---

## 💻 Code Showcase & Model Training

### 1. Basic Tensor Operations & Automated Differentiation

```python
import forge as fg

# Initialize tensors with gradient tracking
x = fg.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = fg.tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
b = fg.tensor([0.1, -0.1], requires_grad=True)

# Forward pass: Linear projection + GELU activation
y = fg.matmul(x, w) + b
loss = fg.sum(fg.nn.functional.gelu(y))

# Backward pass (Autograd)
loss.backward()

print("Loss Value :", loss.numpy())
print("Gradient dL/dw :\n", w.grad.numpy())
```

---

### 2. Live Character-Level Transformer (NanoGPT)

Train a complete causal autoregressive transformer directly on your local device:

```python
import forge as fg
import forge.nn as nn
from forge.models.nanogpt import GPT, GPTConfig

config = GPTConfig(block_size=32, vocab_size=64, n_layer=4, n_head=4, n_embd=64, bias=False)
model = GPT(config)
optimizer = fg.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(40):
    optimizer.zero_grad()
    logits = model(input_tokens)
    loss = criterion(logits, target_tokens)
    loss.backward()
    optimizer.step()
```

---

### 3. Speech-to-Text & Acoustic Signal Processing (`forge.fft` + `forge.nn`)

Extract real Fourier Mel-spectrograms from raw 16kHz audio waveforms:

```python
import forge as fg
import forge.nn as nn

# Fast Fourier Transform (Complex Spectrum)
fft_complex = fg.fft.rfft(raw_audio, n=1024, dim=-1)

# Power Spectrogram Energy
power_spec = (fft_complex.real.pow(2.0) + fft_complex.imag.pow(2.0) + 1e-6).log()

# 1D Convolutional Audio Feature Extractor
conv = nn.Conv1d(in_channels=513, out_channels=64, kernel_size=3, padding=1)
audio_features = conv(power_spec)
```

---

## 🧩 Comprehensive Module Directory

| Module | Core Functionality | Key Operators / Classes |
| :--- | :--- | :--- |
| **`forge`** | Core Tensor Engine & Factories | `tensor`, `zeros`, `ones`, `randn`, `matmul`, `einsum`, `reshape`, `permute`, `where` |
| **`forge.nn`** | Deep Learning Layers & Containers | `Linear`, `Conv1d`, `Conv2d`, `MultiheadAttention`, `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `Embedding`, `CrossEntropyLoss`, `MSELoss` |
| **`forge.optim`** | Optimizers & Rate Schedulers | `SGD`, `Adam`, `AdamW`, `RMSprop`, `CosineAnnealingLR`, `StepLR` |
| **`forge.linalg`** | Linear Algebra Decomposition | `norm`, `svd`, `qr`, `cholesky`, `inv`, `pinv`, `det`, `matrix_rank`, `solve`, `eigh` |
| **`forge.fft`** | Discrete Fourier Transforms | `rfft`, `irfft`, `fft`, `ifft`, `fft2`, `ifft2`, `rfft2`, `irfft2`, `fftfreq`, `fftshift` |
| **`forge.special`** | Transcendental & Error Functions | `erf`, `erfc`, `erfinv`, `gammaln`, `digamma`, `expm1`, `log1p`, `expit`, `logit`, `sinc`, `i0`, `xlogy` |
| **`forge.distributions`**| Probability Distributions & KL | `Normal(rsample)`, `Uniform`, `Bernoulli`, `Categorical`, `kl_divergence` |
| **`forge.models`** | Pre-architected Reference Models | `GPT`, `GPTConfig`, `LLaMA` |

---

## 🌍 The AMEVA Foundation (아메바 재단)

**AMEVA-Forge** is an open-source deep learning initiative created and governed by the **AMEVA Foundation** (아메바 재단).

### 🎯 Our Mission & Zero-Server Paradigm
The AMEVA Foundation is dedicated to the **democratisation of client-side artificial intelligence and edge computing**. We envision a decentralized web where deep learning training, fine-tuning, and neural inference occur directly on user devices (browsers, laptops, tablets, and smartphones)—eliminating massive cloud server costs, safeguarding user data sovereignty with zero network leakage, and providing zero-latency neural intelligence everywhere.

### 🌟 Core Foundation Initiatives
- 🌐 **AMEVA-Forge Tensor Engine**: High-performance WebGPU tensor computation & reverse-mode autograd engine.
- 🎙️ **On-Device Whisper STT**: Real-time acoustic FFT signal processing & speech recognition running locally in browser.
- 🤖 **Decentralized LLM Transformers**: Zero-install character and subword transformer models with zero cloud dependency.
- 🔒 **Privacy-First AI Sovereignty**: Sovereign on-device machine learning where private data never leaves client silicon.

* **Foundation Portal**: [https://uno-km.github.io/ameva-forge/](https://uno-km.github.io/ameva-forge/)
* **Official Repository**: [https://github.com/uno-km/ameva-forge](https://github.com/uno-km/ameva-forge)
* **Community & Governance**: [https://github.com/uno-km/ameva-forge/issues](https://github.com/uno-km/ameva-forge/issues)
* **Sponsorship & Research Partnerships**: [https://github.com/uno-km](https://github.com/uno-km)

---

## 📜 Quality Assurance & Verification

Every release of AMEVA-Forge undergoes rigorous multi-tier verification:
* **292 Unit & Stress Tests**: 100% automated pass rate across CPU, GPU fallback, mathematical accuracy, and memory quota managers.
* **Finite-Difference Gradcheck**: Numerical gradient validation against analytical Vector-Jacobian backward formulations.
* **Memory Lifecycle Audit**: Zero-leak allocation token reclamation and buffer recycling verification across 10,000+ continuous execution cycles.

---

## 📄 License & Citation

AMEVA-Forge is licensed under the [MIT License](LICENSE).

```bibtex
@software{ameva_forge_2026,
  author = {AMEVA Foundation},
  title = {AMEVA-Forge: High-Performance WebGPU-Accelerated Tensor Computation Engine},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/uno-km/ameva-forge}
}
```
