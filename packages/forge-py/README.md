# AMEVA-Forge (`ameva-forge`)

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/ameva-forge/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/ameva-forge/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/uno-km/ameva-forge/blob/main/LICENSE)
[![Test Suite](https://img.shields.io/badge/tests-292%2F292%20passed%20(100%25)-brightgreen.svg)](https://github.com/uno-km/ameva-forge)
[![WebGPU Acceleration](https://img.shields.io/badge/acceleration-WebGPU%20WGSL%20%2B%20CPU-orange.svg)](https://uno-km.github.io/ameva-forge/)

**High-Performance Client-Side Tensor Computation Engine & Reverse-Mode Autograd Framework Powered by WebGPU.**

Developed and maintained by the **AMEVA Foundation** (아메바 재단), AMEVA-Forge is an industrial-grade, zero-server-cost deep learning library engineered to execute high-throughput tensor operations, automated differentiation, and end-to-end neural model training natively within client runtimes (WebGPU, WASM/Pyodide, and native Python environments).

---

## Architectural Pillars

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

1. **Deterministic Autograd & Topological Execution**  
   Strict reverse-mode automatic differentiation graph with cycle detection, multi-output tuple bindings, in-place version invalidation, and scalar-tensor memory optimization.
2. **WebGPU Hardware Acceleration**  
   Direct-to-silicon WGSL compute shaders featuring 8-dimensional non-contiguous stride dispatching, 2D workgroup partitioning ($65,535 \times 65,535$), and explicit buffer lifecycle tracking.
3. **PyTorch 1:1 API Parity**  
   Seamless drop-in compatibility across neural layers (`nn.Module`, `nn.MultiheadAttention`, `nn.Conv2d`), mathematical primitives (`linalg`, `fft`, `special`), and probabilistic graphical models (`distributions`).
4. **Zero-Server Infrastructure (Edge & Browser)**  
   Execute full model fine-tuning and inference directly inside the browser using Pyodide and WebGPU with zero cloud compute cost and total data privacy.

---

## Installation

Install the official package from PyPI:

```bash
pip install ameva-forge
```

Or install from source with development dependencies:

```bash
git clone https://github.com/uno-km/ameva-forge.git
cd ameva-forge/packages/forge-py
pip install -e .
```

---

## Quick Start

### 1. Basic Tensor & Automated Differentiation

```python
import forge as fg

# Initialize tensors with gradient tracking
x = fg.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = fg.tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
b = fg.tensor([0.1, -0.1], requires_grad=True)

# Forward pass: Linear projection + GELU activation
y = fg.matmul(x, w) + b
loss = fg.sum(fg.nn.functional.gelu(y))

# Compute Vector-Jacobian Products (Autograd backward)
loss.backward()

print("Loss Value :", loss.numpy())
print("Gradient dL/dw :\n", w.grad.numpy())
```

---

### 2. Character-Level Transformer (NanoGPT)

Train a complete causal autoregressive transformer directly on your local device:

```python
import forge as fg
import forge.nn as nn
from forge.models.nanogpt import GPT, GPTConfig

# Define model configuration
config = GPTConfig(
    block_size=32,
    vocab_size=64,
    n_layer=4,
    n_head=4,
    n_embd=64,
    bias=False
)
model = GPT(config)
optimizer = fg.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

# Training step (Batch Size: 8, Sequence Length: 32)
input_tokens = fg.tensor([[1, 5, 12, 3]], dtype="int32")
target_tokens = fg.tensor([[5, 12, 3, 18]], dtype="int32")

optimizer.zero_grad()
logits = model(input_tokens)
loss = criterion(logits, target_tokens)
loss.backward()
optimizer.step()
```

---

### 3. Speech-to-Text & Acoustic Signal Processing (`forge.fft` + `forge.nn`)

Compute real Fourier Mel-spectrograms from raw acoustic waveforms:

```python
import forge as fg
import forge.nn as nn

# 16kHz PCM audio waveform (Batch: 4, Samples: 8000)
raw_audio = fg.tensor(audio_data, dtype="float32")

# Fast Fourier Transform (Complex Spectrum)
fft_complex = fg.fft.rfft(raw_audio, n=1024, dim=-1)

# Power Spectrogram Energy
power_spec = (fft_complex.real.pow(2.0) + fft_complex.imag.pow(2.0) + 1e-6).log()

# 1D Convolutional Audio Feature Extractor
conv = nn.Conv1d(in_channels=513, out_channels=64, kernel_size=3, padding=1)
audio_features = conv(power_spec)
```

---

## Comprehensive Module Directory

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

## In-Browser Zero-Install Execution (WebGPU + Pyodide)

AMEVA-Forge packages a single bundled JavaScript distribution (`forge-py-bundle.js`) that mounts into browser-native Pyodide runtimes:

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script src="https://uno-km.github.io/ameva-forge/dist/forge-py-bundle.js"></script>
<script>
  async function runClientDeepLearning() {
    let pyodide = await loadPyodide();
    await window.loadAmevaForgeBundle(pyodide);
    
    await pyodide.runPythonAsync(`
      import forge as fg
      x = fg.randn((1024, 1024), device="gpu")
      y = fg.matmul(x, x)
      print("Computed 1024x1024 on WebGPU Hardware:", y.shape)
    `);
  }
  runClientDeepLearning();
</script>
```

---

## The AMEVA Foundation (아메바 재단)

**AMEVA-Forge** is an open-source initiative directed by the **AMEVA Foundation** (아메바 재단).

### Our Mission
The AMEVA Foundation is dedicated to the **democratisation of client-side artificial intelligence**. We envision a decentralized web where deep learning inference, fine-tuning, and scientific computation occur directly on user devices—eliminating centralized server costs, safeguarding user data sovereignty, and providing zero-latency neural capabilities everywhere.

* **Foundation Portal**: [https://uno-km.github.io/ameva-forge/](https://uno-km.github.io/ameva-forge/)
* **Official Repository**: [https://github.com/uno-km/ameva-forge](https://github.com/uno-km/ameva-forge)
* **Issue Tracker & Governance**: [https://github.com/uno-km/ameva-forge/issues](https://github.com/uno-km/ameva-forge/issues)

---

## Quality Assurance & Verification

Every release of AMEVA-Forge undergoes rigorous multi-tier verification:
* **292 Unit & Stress Tests**: 100% automated pass rate across CPU, GPU fallback, mathematical accuracy, and memory quota managers.
* **Finite-Difference Gradcheck**: Numerical gradient validation against analytical Vector-Jacobian backward formulations.
* **Memory Lifecycle Audit**: Zero-leak allocation token reclamation and buffer recycling verification across 10,000+ continuous execution cycles.

---

## License & Citation

AMEVA-Forge is licensed under the [MIT License](https://github.com/uno-km/ameva-forge/blob/main/LICENSE).

```bibtex
@software{ameva_forge_2026,
  author = {AMEVA Foundation},
  title = {AMEVA-Forge: High-Performance WebGPU-Accelerated Tensor Computation Engine},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/uno-km/ameva-forge}
}
```

