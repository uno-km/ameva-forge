# @ameva/forge

> **High-Performance Browser-Native Tensor Computation Engine & Reverse-Mode Autograd Framework Powered by WebGPU.**

[![npm version](https://img.shields.io/npm/v/@ameva/forge?color=CB3837&logo=npm&logoColor=white&label=npm)](https://www.npmjs.com/package/@ameva/forge)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![WebGPU Acceleration](https://img.shields.io/badge/Acceleration-WebGPU%20WGSL%20%2B%20Pyodide-orange.svg)](https://uno-km.github.io/ameva-forge/)
[![Tests](https://img.shields.io/badge/Tests-292%2F292%20Passing%20(100%25)-brightgreen.svg)](https://github.com/uno-km/ameva-forge)

An industrial-grade, zero-server-cost deep learning runtime engineered to execute high-throughput tensor operations, automatic differentiation, and end-to-end neural model training natively inside client WebGPU and WebAssembly runtimes.

---

## Key Features

1. **Deterministic Autograd & Topological Execution**: Reverse-mode automatic differentiation graph with cycle detection, multi-output tuple bindings, in-place version tracking, and scalar-tensor memory optimization.
2. **WebGPU Hardware Acceleration**: Direct-to-silicon WGSL compute shaders featuring 8-dimensional non-contiguous stride dispatching, 2D workgroup partitioning ($65,535 \times 65,535$), and explicit buffer lifecycle recycling.
3. **PyTorch 1:1 API Parity**: Seamless drop-in compatibility across neural layers (`nn.Module`, `nn.MultiheadAttention`, `nn.Conv2d`), mathematical primitives (`linalg`, `fft`, `special`), and probabilistic models (`distributions`).
4. **Zero-Server Infrastructure**: Execute full model fine-tuning and inference directly inside the browser using Pyodide and WebGPU with zero cloud compute cost and total data privacy.

---

## Installation

```bash
npm install @ameva/forge
```

---

## Quick Start

```typescript
import { Tensor, nn, optim } from '@ameva/forge';

// Initialize WebGPU Tensor Context
const x = new Tensor([1.0, 2.0, 3.0, 4.0], { shape: [2, 2], requiresGrad: true });
const w = new Tensor([0.5, -0.5, 1.0, 2.0], { shape: [2, 2], requiresGrad: true });

// Forward Pass
const y = x.matmul(w).relu().sum();

// Reverse-Mode Automatic Differentiation
y.backward();

console.log('Output Value:', y.item());
console.log('Gradients of x:', x.grad.toArray());
```

---

## Architecture

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

## License

MIT License. Copyright (c) 2026 uno-km (AMEVA Foundation).
