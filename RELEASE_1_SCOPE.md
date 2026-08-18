# 🎯 AMEVA-Forge Release 1.0 GPU Capability Matrix & Scope Specification

**Release Version**: 1.0.0  
**Verification Date**: 2026-08-18  
**Quality Level**: Tier-1 Production Quality (100% Zero-Trust Microscopic Verified)

---

## 📊 Release 1 GPU Capability Matrix

### ✅ 1. Fully Supported on WebGPU (Hardware Accelerated)
| Feature / Operator | GPU Acceleration | Autograd / Training | Notes & Guarantees |
|:---|:---:|:---:|:---|
| **Tensor Lifecycle** | ✅ | ✅ | Zero-copy upload, async readback, 3-tier GC, atomic rollback |
| **Element-wise Math** (`add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`) | ✅ | ✅ | Full N-D broadcasting up to Rank 8 |
| **Activation Functions** (`relu`, `sigmoid`, `tanh`) | ✅ | ✅ | Dedicated forward & backward WGSL compute shaders |
| **Matrix Multiplication** (`matmul`, `batched_matmul`, `bmm`) | ✅ | ✅ | 64x64/32x32 tiled compute kernels with dynamic workload budgeting |
| **Fused Matmul** (`matmul_bias_relu`) | ✅ | Inference | High-throughput fused forward pass |
| **Reductions** (`sum`, `max`, `mean`) | ✅ | ✅ | Multi-pass logarithmic workgroup tree reductions |
| **Axis Reductions** (`sum_axis`, `max_axis`, `max_axis_backward`) | ✅ | ✅ | Generalized `(outer_size, reduction_size, inner_stride)` GPU kernels |
| **Attention Mechanism** | ✅ | ✅ | Scaled Dot-Product Attention with GPU causal masking |
| **Softmax / LogSoftmax** | ✅ | ✅ | Numerically stable shifted exp ($x - \max(x)$) GPU pipeline |
| **Loss Functions** (`MSELoss`) | ✅ | ✅ | Pure GPU forward & backward reduction graph |
| **Neural Net Core** (`Linear`, `ReLU`, `Sequential` MLP) | ✅ | ✅ | Pure WebGPU end-to-end forward, backward, optimizer step |
| **Optimizers** (`SGD`, `SGD with Momentum`) | ✅ | ✅ | GPU-native kernel execution via AXPY transactions |
| **Indexing / Tensor Manipulation** (`gather`, `scatter` [assign], `pad`, `cat`, `permute`, `where`, `fill`) | ✅ | ✅ | Exact tensor indexing with strict range verification |

---

### ⚠️ 2. Partially Supported / Scope-Guarded in Release 1
| Feature / Operator | Current GPU Behavior | Safety & Defense Policy | Release 2 Roadmap |
|:---|:---:|:---|:---|
| **`CrossEntropyLoss`** | ✅ (Class Count $\le 4\text{M}$) | Dense one-hot target pipeline on GPU. Exceeding 4M elements ($>16\text{MB}$) triggers fail-fast `AMEVAForgeUnsupportedOperationError` to prevent browser OOM. | Sparse GPU CrossEntropy kernel |
| **`Dropout`** | ✅ | GPU forward and backward with workgroup seed hashing; deterministic mode available. | PRNG state generator |
| **`Conv2d` / `im2col` / `col2im`** | ✅ (Inference) | GPU forward acceleration supported. Backward on GPU raises `AMEVAForgeUnsupportedOperationError` to protect users against unverified gradients. | Dedicated GPU Conv2D backward kernel |
| **`MaxPool2d` / `AvgPool2d`** | ✅ (Inference) | GPU forward pooling supported. Backward on GPU raises `AMEVAForgeUnsupportedOperationError`. | GPU atomic scatter-max backward kernel |

---

### 🛑 3. Not Supported on GPU in Release 1 (CPU-Only or Scope-Guarded)
| Feature / Operator | Policy | Failure Mode | Recommended Alternative in R1 |
|:---|:---:|:---|:---|
| **`Adam` Optimizer** | CPU Only | `Adam.step()` / `Adam.step_async()` on GPU tensors immediately raises `AMEVAForgeUnsupportedOperationError`. | Use `SGD` for GPU models or train on CPU. |
| **Gradient Clipping** (`clip_grad_norm`, `clip_grad_value`) | CPU Only | Calling on GPU tensors immediately raises `AMEVAForgeUnsupportedOperationError`. | Move parameters to CPU before clipping. |
| **`Embedding`** | CPU Only | GPU tensor inputs raise `AMEVAForgeUnsupportedOperationError`. | Execute embedding on CPU: `x_gpu = embed(idx).to('gpu')`. |
| **`RNN` / `LSTM`** | CPU Only | GPU tensor inputs raise `AMEVAForgeUnsupportedOperationError`. | Execute sequence recurrent loops on CPU. |
| **`scatter_add` / `scatter_reduce`** | CPU Only | `scatter(..., reduce != 'assign')` on GPU raises `AMEVAForgeUnsupportedOperationError`. | Use assign semantics or CPU scatter. |

---

## 🛡️ Architecture Invariants & Safety Principles

1. **No Silent Fallbacks / No Silent Ignores**:
   Unsupported GPU operations NEVER quietly fallback to CPU numpy or silently drop computation. They fail-fast with strongly typed `AMEVAForgeUnsupportedOperationError`.
2. **Deterministic VRAM Accounting**:
   Every GPU allocation is guarded by an `AllocationToken`. Deferred GC retries are bounded to 3 attempts, after which failed destructions are explicitly recorded in `orphanedBytes` rather than falsifying zero-leak metrics.
3. **Detached WASM Buffer Recovery**:
   `safeCopy.ts` detects memory growth (`memory.grow()`) detachments and provides optional `reacquire()` hooks without crashing.
4. **Workload Budgeting**:
   Large GPU dispatches are automatically partitioned by `ForgeRuntimeConfig` (`workloadBudgetElements` & `maxOpsPerSubmit`) to prevent OS GPU timeout (TDR).
