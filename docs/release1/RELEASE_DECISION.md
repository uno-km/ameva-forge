# Release 1 Decision

**Date**: 2026-08-13
**Branch**: `release1/hardening-20260813`
**Evaluator**: Release 1 Hardening — Claude Code

## 판정: Internal Alpha 유지

### 근거

설계서 07 (Master Prompt) 최종 판정 규칙에 따라:

> "browser MLP 또는 memory gate가 실패/미실행이면 Internal Alpha 유지다."

현재 상태:
- ✅ P0 Core Architecture 변경 완료 (async executeGraph, allocator 통합, error scope await)
- ✅ P0 전역 error 채널 제거
- ✅ P0 PyProxy cleanup
- ❌ P0 Browser MLP Training Test **미실행** (환경 제약)
- ❌ P0 Python async bridge 런타임 검증 **미완료** (Pyodide 환경 필요)
- ❌ P1 1,000-step Memory Test **미실행**
- ❌ P1 CPU/GPU Forward Parity **미검증**

### P0 Gate 상태

| Gate | 상태 | 비고 |
|------|------|------|
| executeGraph가 commit 전 error scope 확인 | ✅ PASS | await 3회 |
| error scope 결과 → Python typed exception | ✅ PASS | _map_js_error |
| 실패 graph rollback | ✅ PASS | createdHandles + paramsAllocations |
| direct device.createBuffer 제거 | ✅ PASS | graphExecutor + gpuCore 모두 allocateBuffer |
| PyProxy/JsProxy finally cleanup | ✅ PASS | upload bufProxy, bridge.py |
| 실제 Chrome MLP loss 감소 | ❌ 미실행 | E2E 환경 필요 |
| GPU validation error 0건 | ❌ 미실행 | E2E 환경 필요 |

### 완료된 작업 요약

- 11 test suites, 68 unit tests ALL PASS
- Build 성공 (rollup → dist/index.js + dist/index.esm.js)
- Python compileall 성공
- 3 new test files (graphTransaction, errorPropagation, resourceLifecycle)
- 5 new TS error types + 6 new Python exception types
- package.json scripts 4개 추가

### RC 판정 요건

`0.1.0-rc.1` candidate로 진행하려면:
1. E2E 환경 구성 (Chrome + Pyodide + WebGPU)
2. 2-layer MLP 50+ step loss 감소 확인
3. 1,000-step memory baseline 복귀 확인
4. GPU validation error 0건 확인
5. Core forward CPU/GPU parity 확인
6. Documentation 정정 완료

이 조건이 모두 충족될 때까지 **Internal Alpha** 상태를 유지합니다.
