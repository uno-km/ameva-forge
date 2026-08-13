# Release Blockers (Red Team Review 2)

## ID-01: reshape Graph Contract
- **Title**: `reshape` operation crashes on GPU execution.
- **Severity**: Critical
- **Status**: Confirmed
- **Evidence**: `ops.py` L629 sends `op='reshape'`, but `ALLOWED_OPS` in `graphExecutor.ts` L55 does not include `reshape`.
- **Affected Files**: `forge-py/src/forge/ops.py`, `forge/src/tensor/graphExecutor.ts`
- **User Impact**: Any call to `.reshape()`, `.unsqueeze()`, `.squeeze()` on a GPU tensor results in `AMEVAForgeSecurityError`.
- **Reproduction**: Run `x.reshape((...))` and `realize()`.
- **Release 1 Impact**: Must fix to release Release 1.
- **Fix Scope**: Add metadata-only view support in TS runtime or kernel copy.
- **Completion Criteria**: E2E shape operations pass on WebGPU.

## ID-02: sum/max Reduction Correctness
- **Title**: `sum` and `max` operations perform partial reduction (workgroup only).
- **Severity**: Critical
- **Status**: Confirmed
- **Evidence**: `sum.wgsl.ts` and `graphExecutor.ts` (L512). The loop `while(currentSize > 1)` uses output `byteLength` (which is 4 bytes for scalar output), so it never executes multi-stage passes. It simply copies the first workgroup's partial result.
- **Affected Files**: `forge/src/tensor/graphExecutor.ts`, `forge/src/tensor/kernels/sum.wgsl.ts`
- **User Impact**: `sum()`, `mean()`, and loss functions produce incorrect values for tensors > 256 elements.
- **Reproduction**: Sum a tensor of 512 ones, result will be 256 instead of 512.
- **Release 1 Impact**: Must fix for math parity.
- **Fix Scope**: Calculate initial `currentSize` from input shape, not output byteLength.
- **Completion Criteria**: Numerical parity with CPU for large reductions (>65536 elements).

## ID-03: GPU Validation Error Propagation
- **Title**: Unawaited GPU validation errors allow invalid graph success.
- **Severity**: High
- **Status**: Confirmed
- **Evidence**: `graphExecutor.ts` L680 calls `void device.popErrorScope().then(...)` without awaiting, returning handles immediately.
- **Affected Files**: `forge/src/tensor/graphExecutor.ts`
- **User Impact**: Failures in GPU submission yield garbage output handles silently.
- **Reproduction**: Submit a malformed op, handles are still returned to Python.
- **Release 1 Impact**: Must fix for stability.
- **Fix Scope**: Convert `executeGraph` to be fully asynchronous or correctly handle promises before resolving handles.
- **Completion Criteria**: A GPU validation error propagates to Python as an exception.

## ID-04: Quota Ledger Bypass
- **Title**: Temporary buffers and duplicate releases bypass quota limits.
- **Severity**: High
- **Status**: Confirmed
- **Evidence**: `graphExecutor.ts` L519 directly calls `device.createBuffer` bypassing `_globalQuotaManager`. `quota.ts` L87 clamps to 0, allowing duplicate releases to offset other live allocations.
- **Affected Files**: `forge/src/tensor/graphExecutor.ts`, `forge/src/webgpu/quota.ts`
- **User Impact**: VRAM exhaustion due to untracked allocations and offset bypass.
- **Release 1 Impact**: Must fix for resource safety.
- **Fix Scope**: Tie allocation tokens explicitly to releases, enforce tracking on temporary buffers.
- **Completion Criteria**: Duplicate releases throw, and all temporary buffers are tracked.

## ID-05: globalThis raw GPUDevice Exposure
- **Title**: Raw GPUDevice is exposed to third-party scripts.
- **Severity**: Medium
- **Status**: Confirmed
- **Evidence**: `device.ts` L31 exposes `(globalThis as any).__AMEVA_DEVICE__ = device;`.
- **Affected Files**: `forge/src/webgpu/device.ts`
- **User Impact**: Malicious scripts in the same realm can access and destroy the GPU device.
- **Release 1 Impact**: Should hide raw device.
- **Fix Scope**: Encapsulate device within module scope.
- **Completion Criteria**: `globalThis.__AMEVA_DEVICE__` is undefined.

## ID-06: Python Test Harness Wrapped stdout
- **Title**: Windows pytest crashes due to hardcoded stdout wrapping.
- **Severity**: High
- **Status**: Confirmed
- **Evidence**: `test_run_all.py` L11 wraps `sys.stdout = io.TextIOWrapper(...)`, breaking `pytest` capture.py.
- **Affected Files**: `forge-py/tests/test_run_all.py`
- **User Impact**: Inability to run test suite locally on Windows with pytest.
- **Release 1 Impact**: Required for CI/CD integrity.
- **Fix Scope**: Remove explicit wrapping or condition it out of pytest runners.
- **Completion Criteria**: `pytest` executes tests without `I/O operation on closed file`.
