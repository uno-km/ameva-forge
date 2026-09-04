# AMEVA-Forge (`ameva-forge`)

<div align="center">

[![Official Documentation](https://img.shields.io/badge/docs-uno--km.vercel.app%2Flib%2Fforge-004499?style=for-the-badge&logo=vercel)](https://uno-km.vercel.app/lib/forge/)
[![PyPI version](https://img.shields.io/pypi/v/ameva-forge.svg?style=for-the-badge&color=007ec6&logo=pypi&logoColor=white)](https://pypi.org/project/ameva-forge/)
[![NPM version](https://img.shields.io/npm/v/@uno-km/ameva-forge.svg?style=for-the-badge&color=cb3837&logo=npm&logoColor=white)](https://www.npmjs.com/package/@uno-km/ameva-forge)
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

### Python (PyPI Official Distribution)
```bash
pip install ameva-forge
```

### Modern Web Browser (WebGPU Universal Plug & Play SDK)
Embed the zero-server WebGPU runtime directly into your web application:

```html
<!-- Load AMEVA-Forge WebGPU Runtime Engine -->
<script src="https://uno-km.vercel.app/lib/forge/dist/index.js"></script>
```

Or install via NPM / GitHub Packages:
```bash
npm install @uno-km/ameva-forge
```

---

## 🏛️ Architectural Overview

AMEVA-Forge connects a deterministic Python autograd frontend and high-level Plug & Play Model Ingestion directly to client-side WebGPU WGSL compute shaders:

```text
+-----------------------------------------------------------------------------------+
|                        AMEVA-Forge Plug & Play Model Hub                          |
|   Hugging Face GGUF Direct URL  |  Local File Drag & Drop  |  OPFS/Cache Storage  |
+-----------------------------------------------------------------------------------+
|                     Universal Neural Runtime & Tokenizer Engine                   |
|   Byte-Level BPE / SentencePiece  *  Top-K/Top-P Sampler  *  Inference Web Worker  |
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

1. **Plug & Play On-Device Model Hub (Zero Server Cost)**  
   Mount external GGUF models (`SmolLM-135M`, `Qwen2.5-0.5B`, `LLaMA-3.2-1B`) directly into browser VRAM via Hugging Face CDN URL or drag & drop with zero cloud server compute costs.

2. **Universal Byte-Level BPE & SentencePiece Tokenizer**  
   Built-in reversible tokenizer (`BPETokenizer`) extracting vocabulary and BPE rules directly from GGUF metadata for 100% loss-less text encoding and decoding.

3. **Web Worker Background Inference (60 FPS Non-Blocking)**  
   Runs heavy neural decode loops and FlashAttention inside dedicated Web Workers, completely preventing browser UI freezing and OS GPU TDR timeouts.

4. **WebGPU Hardware-Accelerated WGSL Compute Pipeline**  
   Custom WGSL compute shaders for fused matrix multiplication, FlashAttention, RMSNorm, SwiGLU, tensor reduction, element-wise broadcasting, convolutions, LayerNorm, and Softmax operating directly on client GPU hardware.

5. **Deterministic Autograd & Topological Execution**  
   Strict reverse-mode automatic differentiation graph with cycle detection, multi-output tuple bindings, in-place version invalidation, and scalar-tensor memory optimization.

6. **Zero-Copy Buffer Recycling & Memory Token Pools**  
   Direct GPU buffer lifecycle management with zero memory leaks, reusable staging buffers, and asynchronous queue dispatching.

---

## 🎮 Plug & Play Browser JavaScript Example

```javascript
import { ModelLoader } from '@uno-km/ameva-forge';

// 1. Mount external GGUF model into WebGPU VRAM directly from Hugging Face
const session = await ModelLoader.loadModel(
  'https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/resolve/main/smollm-135m-instruct-q4_k_m.gguf',
  { onProgress: (p) => console.log(`[Loading] ${p.percentage}%: ${p.statusText}`) }
);

// 2. Stream autoregressive text generation at 60 FPS
await session.prompt('Explain WebGPU in simple terms', {
  maxNewTokens: 64,
  temperature: 0.7,
  onToken: (chunk, progress) => {
    process.stdout.write(chunk); // Real-time typewriter output
  }
});
```

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
