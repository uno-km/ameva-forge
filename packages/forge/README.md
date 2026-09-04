# @ameva/forge (WebGPU On-Device AI Engine)

> **Universal Browser-Native WebGPU Deep Learning Runtime & Plug-and-Play On-Device Model Hub.**

[![npm version](https://img.shields.io/npm/v/@ameva/forge?color=CB3837&logo=npm&logoColor=white&label=npm)](https://www.npmjs.com/package/@ameva/forge)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![WebGPU Native](https://img.shields.io/badge/WebGPU-Pure_WGSL_Shaders-blueviolet.svg)](https://uno-km.vercel.app/lib/forge/)
[![Tests](https://img.shields.io/badge/Tests-279%2F279_Passing_(100%25)-brightgreen.svg)](https://uno-km.vercel.app/lib/forge/benchmarks.html)

**@ameva/forge** is an industrial-grade, client-side WebGPU neural execution runtime designed specifically for modern web browsers. It empowers web applications to run state-of-the-art neural networks locally on user GPUs with **zero server costs**, **100% data privacy**, and **zero-latency on-device inference**.

---

## ⚡ Key Highlights for Web Developers

1. **Plug & Play GGUF Model Hub**  
   Load quantized models (LLaMA-3, SmolLM, Qwen2.5) directly from Hugging Face URLs or via local drag-and-drop. Automatically bypasses 32-bit WASM 2GB limits using Direct DMA buffer mapping.
2. **Built-in Byte-Level BPE & SentencePiece Tokenizer**  
   Full client-side tokenization engine that reconstructs vocabularies from GGUF metadata. Reversible, lossless text encoding and decoding with zero external dependencies.
3. **60 FPS Non-Blocking Autoregressive Streaming**  
   Generate LLM text with real-time typewriter effects without freezing the UI thread, powered by cooperative event-loop scheduling.
4. **Web Worker Thread Isolation**  
   Offload heavy transformer decoding and GPU dispatches to dedicated Web Workers to ensure perfectly smooth animations and prevent OS GPU timeout (TDR) crashes.
5. **Universal Multimodal Runtimes**  
   Hardware-accelerated compute shaders for STT (Whisper Mel-STFT), TTS (Waveform synthesis), Vision (CLIP ViT), and Diffusion (VAE latent decoding).

---

## 📦 Installation

### NPM / Yarn / PNPM
```bash
npm install @ameva/forge
# or
yarn add @ameva/forge
```

### Browser Direct CDN (Zero-Install)
```html
<script src="https://uno-km.vercel.app/lib/forge/dist/index.js"></script>
```

---

## 🚀 Quick Start (TypeScript / JavaScript)

### 1. Plug & Play LLM Inference from Hugging Face

```typescript
import { ModelLoader, LLMTextGenerator } from '@ameva/forge';

// 1. Stream & Cache GGUF model directly from Hugging Face CDN
const loader = new ModelLoader({ cacheStorage: true });
const model = await loader.loadFromUrl(
  'https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/resolve/main/smollm-135m-instruct.q4_k_m.gguf',
  (progress) => console.log(`Downloading: ${(progress * 100).toFixed(1)}%`)
);

// 2. Initialize 60 FPS non-blocking streaming generator
const generator = new LLMTextGenerator(model, {
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repetitionPenalty: 1.1
});

// 3. Generate streaming tokens
await generator.generateStream("Explain WebGPU in one sentence:", (tokenText) => {
  document.getElementById('output').textContent += tokenText;
});
```

---

### 2. Low-Level WebGPU Tensor Autograd

```typescript
import { Tensor, initWebGPU } from '@ameva/forge';

// Initialize WebGPU adapter & device context
await initWebGPU();

// Create tensors with gradient tracking
const x = new Tensor([1.0, 2.0, 3.0, 4.0], { shape: [2, 2], requiresGrad: true });
const w = new Tensor([0.5, -0.5, 1.0, 2.0], { shape: [2, 2], requiresGrad: true });

// Forward pass through native WGSL compute shaders
const y = x.matmul(w).relu().sum();

// Reverse-mode automatic differentiation
await y.backward();

console.log('Output Value:', y.item());
console.log('Gradients of x:', await x.grad.toArray());
```

---

## 🏛️ Multimodal Architecture

```text
+-----------------------------------------------------------------------------------+
|                        Browser Web Application Layer                              |
|   HTML5 Canvas  |  Web Audio API  |  DOM Typewriter UI  |  Drag-and-Drop Dropzone |
+-----------------------------------------------------------------------------------+
|                 @ameva/forge Client Neural Runtime Engine                         |
|   Byte-Level BPE Tokenizer  *  Shifted Softmax Sampler  *  Worker Isolation Bridge|
+-----------------------------------------------------------------------------------+
|                     On-Device GGUF Direct DMA Loader                              |
|       Hugging Face Chunk Streamer  *  OPFS Cache  *  Tensor Name Mapper           |
+-----------------------------------------------------------------------------------+
|                      WebGPU Native Hardware Shaders                               |
|   Tiled MatMul (WGSL)  *  FlashAttention-2  *  Mel STFT FFT  *  VAE GroupNorm     |
+-----------------------------------------------------------------------------------+
```

---

## 📄 License

Apache-2.0 License. Copyright (c) 2026 uno-km (AMEVA Foundation).

