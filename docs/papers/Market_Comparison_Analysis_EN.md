> [!NOTE]
> **Preliminary Analysis** — This competitive analysis is based on publicly available information and has not been independently verified.

# AMEVA WebGPU-Python Bridge vs. Market WebGPU Projects (Comparative Analysis)

This document provides a highly rigorous and objective comparative analysis between existing commercial/open-source WebGPU projects in browser and Python environments and our custom-developed **`AMEVA WebGPU-Python Bridge (ameva_tensor)`**.

---

## 1. Market Landscape and Comparative Analysis

### 🔴 1. wgpu-py (Python Native WebGPU)
The most representative Python WebGPU library. It wraps the Rust-based `wgpu-native` binaries to operate in desktop environments.
* **Pros:**
  - Supports WebGPU APIs in desktop Python environments (Windows, Mac, Linux).
  - Excellent compatibility with 3D rendering libraries (e.g., pygfx).
* **Cons:**
  - **Does not operate natively within web browser (Pyodide/WASM) environments.** (Browsers cannot execute C/Rust `.dll` binaries due to sandbox security).
  - Strictly evolved around Graphics pipelines. It lacks highly optimized tensor math algorithms (like Fused Softmax) tailored for LLMs and deep learning.
* **Comparison with AMEVA:** AMEVA is not for "Desktop Python" but is exclusively dedicated as a 'Tensor Compute Bridge' operating within **"Python in the Browser (WASM)"**.

### 🟡 2. TensorFlow.js & ONNX Runtime Web (JavaScript WebGPU Backend)
The absolute giants of the browser AI ecosystem, led by Google and Microsoft. They have recently begun officially supporting WebGPU backends.
* **Pros:**
  - Delivers extreme optimization and performance in the browser JavaScript environment.
  - The most stable and overwhelmingly fast solution for running heavy, pre-trained models on the web.
* **Cons:**
  - **It is not Python.** AI researchers and data scientists write code in Python (PyTorch, Numpy). To use TF.js or ONNX Web, they must rewrite all logic in JavaScript or export pre-trained models to ONNX. It is impossible to "type Python code interactively in the browser and instantly develop AI logic."
* **Comparison with AMEVA:** AMEVA provides the "continuity of Developer Experience (DX)", allowing users to **type Python code inside the browser and instantly invoke the GPU**, without switching to JS.

### 🔵 3. Apache TVM (WebGPU Target)
A deep learning compiler framework that analyzes model code and compiles it down to WebAssembly and WebGPU.
* **Pros:**
  - Achieves extreme hardware-dependent optimization by analyzing the model structure itself and compiling raw WebGPU shaders.
* **Cons:**
  - Moving a Python model to the browser requires a heavy Ahead-of-Time (AOT) compilation pipeline. Dynamically writing scripts and testing them instantly on the fly is structurally very difficult.

---

## 2. In-Depth Dissection of AMEVA WebGPU-Python Bridge

- Supports WebGPU APIs in desktop Python environments (Windows, Mac, Linux).
- **Core Philosophy:** *"Allow Data Scientists to write Python code inside a browser sandbox, and dispatches to the browser's WebGPU for AI workloads (Theoretical Target Model)."*

### ✅ Absolute Strengths (Pros)
> **Scope note (Theoretical Model):** This document describes an architectural direction. Release 1 validates only the small-MLP scope listed in `reports/release1/RELEASE_DECISION.md`. LLM-scale kernels, extreme benchmarks, and production claims are not validated.

# Market Comparison & Competitive Position Analysis (Theoretical Model)

## 1. Competitive Matrix

| Feature | TensorFlow.js | ONNX Runtime Web | WebDNN | wgpu-py | **AMEVA Bridge (Target)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Language** | JS / TS | JS / TS | JS | Python | **Python (Pyodide)** |
| **GPU Acceleration** | WebGL / WebGPU | WebGL / WebGPU | WebGPU | WebGPU (Desktop) | **WebGPU (Browser)** |
| **PyTorch API Parity** | Low (Custom API) | Medium (ONNX Ops) | Low | High (Python) | **High (PyTorch-like)** |
| **Browser Execution** | Native | Native | Native | Impossible | **Native (WASM + FFI)** |

## 2. Competitive Positioning
The AMEVA Bridge aims to target a niche position: **"A Real-Time GPU Accelerator specifically designed for Browser-based Python Users (Theoretical Target Model),"** aiming to bridge gaps in existing browser-based execution tools.

## 3. Autograd & Training Scope Note
Release 1 contains a basic autograd implementation for the supported MLP path. Its correctness and resource stability are not generalized to CNN, attention, or production-scale training.
Our goal is not to defeat TensorFlow.js. **Our goal is to provide a local web OS tensor engine to the millions of AI researchers worldwide who use web browsers like Jupyter Notebooks, empowering them to multiply 550 billion matrices without ever freezing their tab.**
