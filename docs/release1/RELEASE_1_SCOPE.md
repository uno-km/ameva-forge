# AMEVA-Forge Release 1 Scope Contract

**Release Target**: `0.1.0-alpha` / `0.1.0-rc.1`  
**Status**: ACTIVE CODE CONTRACT  

---

## 0. Official Release 1 Objective Statement

> **"지원 브라우저에서 Pyodide Python 코드로 2-Layer MLP를 생성하고, WebGPU에서 forward, backward, SGD update를 반복 실행하며, 수학적 정확성과 GPU 메모리 안정성을 검증 가능한 형태로 제공한다."**

AMEVA-Forge Release 1 is strictly scoped to this objective. It is **NOT** a general-purpose deep learning framework.

---

## 1. Mandatory In-Scope Features

### 1.1 Core Tensor & Lifecycle
- `fg.tensor(data)`
- CPU -> GPU upload (`tensor.to("gpu")`)
- GPU -> CPU readback (`await tensor.numpy_async()`, `await tensor.realize()`)
- Idempotent tensor disposal (`tensor.dispose()`)

### 1.2 Mathematical Operations (CPU, GPU, Autograd, Backward)
- `add` (element-wise addition, scalar broadcast)
- `sub` (element-wise subtraction)
- `mul` (element-wise multiplication)
- `div` (element-wise division)
- `neg` (element-wise negation)
- `matmul` (2D matrix multiplication $M \times K @ K \times N \to M \times N$)
- `transpose` (2D matrix transposition $M \times N \to N \times M$)
- `reshape` (tensor dimension restructuring)
- `sum` (scalar reduction sum)
- `relu` (ReLU forward & backward)
- `mse_loss` (MSE loss forward & backward)
- `axpy` (GPU in-place parameter update: $y \leftarrow \alpha x + y$)

### 1.3 Neural Network & Optimization
- `nn.Module`
- `nn.Linear(in_features, out_features, bias=True)`
- `nn.Sequential(*layers)`
- `nn.ReLU()`
- `nn.MSELoss()`
- `Module.to("gpu")` (in-place registered parameter replacement)
- `SGD(params, lr=0.1)` (`step()` for CPU, `await step_async()` for GPU)

### 1.4 System, Runtime & Safety
- Pyodide WASM bridge
- WebGPU graph compilation & serialization
- VRAM Quota accounting (`QuotaManager`, `AllocationToken`)
- Typed exception propagation (`AMEVAForgeError` hierarchy)
- Browser 2-Layer MLP training gate & 1,000-step memory stability gate

---

## 2. Explicitly Excluded Features (Out of Scope / Experimental)

The following features are strictly **EXCLUDED** from Release 1 public APIs, navigation, and release acceptance:
- `Conv2d`, `MaxPool2d`, `AvgPool2d`, `im2col`, `col2im` (CNN architectures)
- `RNN`, `LSTM`, `Transformer`, `Attention`, `PositionalEncoding`
- `BatchNorm2d`, `LayerNorm`, `Dropout`, `Embedding`
- Vision Center / MNIST playground
- Extreme / Synthetic benchmarks (`massive_cpu`, `vram_crusher`, `machine_gun`)
- User-provided arbitrary WGSL execution
- `float16` & `int32` GPU tensor computation
- Multi-GPU & distributed browser training

---

## 3. Strict Quality & Acceptance Gates

1. **Python Unit Tests**: All tests in `packages/forge-py/tests` must pass with 0 failures.
2. **TypeScript Unit Tests**: All Jest suites in `packages/forge/tests` must pass with 0 failures.
3. **TypeScript Build**: `npx tsc --noEmit` must pass with Exit Code 0 (**STATIC PASS**).
4. **Playwright Spec Collection**: All 9 E2E specs must be collected without import errors (**COLLECTION PASS**).
5. **Contract Drift Prevention**: `py -3 scripts/generate_release1_contracts.py --check` must pass.
6. **Documentation Claim Linter**: `py -3 scripts/lint_unverified_claims.py` must pass with 0 violations.
7. **Browser WebGPU Gates**: Physical WebGPU adapter required for 50-step MLP training & 1,000-step memory stability reports.
