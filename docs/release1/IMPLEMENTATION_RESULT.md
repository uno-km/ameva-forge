# Release 1 Implementation Result

**Date**: 2026-08-13
**Branch**: `release1/hardening-20260813`
**Target**: AMEVA-Forge 0.1.0 Public Technical Preview Candidate

## 변경 요약

### Phase A: Baseline
- 브랜치 `release1/hardening-20260813` 생성
- 환경: Node v24.16.0, TypeScript 5.9.3
- Baseline: Build ✅, 7/8 test suites (36 tests), 1 suite 실패 (sourceContract — @types/node 미등록)

### Phase B-C: executeGraph → async + Transaction

#### `packages/forge/src/tensor/graphExecutor.ts`
- `executeGraph` → `async function` returning `Promise<Record<number, TensorHandle>>`
- GPU error scopes: 3개 `await device.popErrorScope()` — commit 전 확인
- `__ameva_last_gpu_error` 전역 채널 완전 제거
- params buffer (`device.createBuffer`) → `allocateBuffer('uniform', txId)`
- reduction buffer (`device.createBuffer`) → `allocateBuffer('temporary', txId)`
- 실패 시 `freeBuffer(alloc.buffer, alloc.token)` 로 rollback
- Error scope → typed error (`AMEVAForgeValidationError`, `AMEVAForgeOutOfMemoryError`, `AMEVAForgeInternalGPUError`)
- Upload `bufProxy.release()` → `try/finally` 보강

#### `packages/forge/src/errors.ts`
- 5개 신규 error class: `AMEVAForgeValidationError`, `AMEVAForgeOutOfMemoryError`, `AMEVAForgeInternalGPUError`, `AMEVAForgeDeviceLostError`, `AMEVAForgeStaleHandleError`

### Phase D: Python Async Bridge

#### `packages/forge-py/src/forge/bridge.py`
- `js_execute_graph` → `async def` with `await core.executeGraph()`
- `_map_js_error()` — JS error → Python typed exception 매핑
- try/finally Proxy cleanup (js_inputs, result_proxy)

#### `packages/forge-py/src/forge/tensor.py`
- `realize()` → `async def` with `await js_execute_graph()`

#### `packages/forge-py/src/forge/errors.py`
- 6개 신규 exception: `AMEVAForgeValidationError`, `AMEVAForgeOutOfMemoryError`, `AMEVAForgeInternalGPUError`, `AMEVAForgeDeviceLostError`, `AMEVAForgeStaleHandleError`, `AMEVAForgeUnsupportedOperationError`

### Phase E: Allocator 통합
- `gpuCore.ts` `dispatchKernel` params buffer → `allocateBuffer('uniform')`
- `freeBuffer` import 및 cleanup 경로 정리

### Phase F: PyProxy Cleanup
- `graphExecutor.ts` upload 경로 bufProxy → `try/finally`
- `gpuCore.ts` readMappedInto — 이미 try/finally 확인

### Phase G: Tests
- `graphTransaction.test.ts` — async 검증, input validation (7 tests)
- `errorPropagation.test.ts` — 13 error types hierarchy (15 tests)
- `resourceLifecycle.test.ts` — allocator, global error, async contract (6 tests)
- **결과: 11 suites, 68 tests ALL PASS**

### Phase J: Packaging
- `package.json` scripts: `typecheck`, `test:unit`, `test:e2e`, `pack:check` 추가
- `tsconfig.json` types: `@types/node` 추가

## 설계서와 실제 코드의 차이

| 설계서 가정 | 실제 코드 | 대응 |
|------------|----------|------|
| Error types 미존재 | errors.ts에 8개, errors.py에 8개 이미 존재 | 신규 5+6개 추가만 |
| QuotaManager 기본 | AllocationToken + 2-stage release 이미 구현 | 그대로 활용 |
| Version 불일치 | npm/Python 모두 0.1.0 | 변경 불필요 |
| TensorRegistry 단순 | dispose idempotent + markPendingRelease 이미 구현 | pending/committed는 향후 |

## 검증 결과

| Gate | 항목 | 결과 |
|------|------|------|
| Gate A | TypeScript build | ✅ PASS |
| Gate A | Python compileall | ✅ PASS |
| Gate B | Jest unit tests (11 suites, 68 tests) | ✅ ALL PASS |
| Gate B | Error type hierarchy | ✅ 13 types verified |
| Gate B | Async contract verified | ✅ |
| Gate B | Allocator integration verified | ✅ |
| Gate B | Global error channel removed | ✅ |
