# Competitor Research & Differentiation (Red Team Review 2)

## Minimum Competitor Baseline

| Framework | Primary Env | Python First? | Browser WebGPU Training? | API Philosophy | Limitations for Browser Training |
|---|---|---|---|---|---|
| **PyTorch** | Server/Native | Yes | No (WASM exists, but no WebGPU backend out-of-the-box) | Eager + Autograd | Heavy dependencies, requires Python server backend or C++ compilation. |
| **TensorFlow.js** | Browser | No | Yes (WebGL / WebGPU) | Eager + Graph (JS) | Requires JavaScript/TypeScript knowledge. No Python syntax. |
| **ONNX Runtime Web** | Browser | No | No (Inference only) | Inference Graph | Cannot be used for training (Autograd missing). |
| **JAX** | Server/Native | Yes | No | Functional | No native browser/WebGPU backend out-of-the-box. |
| **tinygrad** | Server/Native | Yes | Partial (WebGPU backend exists, but not designed primarily for Pyodide/Browser E2E) | Minimalist PyTorch | WebGPU backend is experimental; primarily desktop focused. |
| **AMEVA-Forge** | Browser (Pyodide) | Yes | Yes (Through Custom WGSL & Autograd) | PyTorch-like OOP | FFI Overhead, lack of mature e2e testing. |

## Differentiation Assessment

- **Defensible**: None (yet). True defensibility requires verifiable benchmarks showing FFI latency is manageable, and parity tests proving mathematical correctness.
- **Potential Differentiation**: "Browser-based Python WebGPU training runtime." This is a rare combination. If E2E correctness, stability, and reasonable performance are verified, this becomes highly defensible for education and local prototyping.
- **Superficial**: "PyTorch-like API." (Many frameworks mimic PyTorch; the API itself is not a moat).
- **Disadvantage**: Heavy JSON-based FFI serialization overhead between Python and TypeScript.

*Note: Claims that AMEVA-Forge is "the only" or "the fastest" are strictly prohibited until supported by reproducible benchmarking.*
