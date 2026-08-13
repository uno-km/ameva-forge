# Release 1 Open Blockers

**Date**: 2026-08-13
**Branch**: `release1/hardening-20260813`

## P0 Blockers

### B1: Browser MLP Training Test 미실행
- **상태**: 미완료 (실제 Chrome/Pyodide/WebGPU 환경 필요)
- **영향**: P0 Gate — 이 테스트 없이는 Release 불가
- **필요 조건**: 
  - Headless Chrome with `--enable-unsafe-webgpu`
  - Pyodide v0.25+ 로드
  - `@ameva/forge` bundle 서빙
  - forge wheel `micropip.install()`
  - 2-layer MLP 50 step 실행 → loss 감소 확인
- **예상 해결**: E2E test 환경 구성 후 `packages/e2e-test/runner.js` 활용

### B2: Python async realize() 호출 체인 검증
- **상태**: compileall 통과하나 실제 Pyodide 런타임에서 미검증
- **영향**: P0 — async realize()의 호출 체인이 Pyodide event loop에서 올바르게 동작하는지 확인 필요
- **세부**: tensor.py 내 `realize()`를 호출하는 모든 경로가 `await`를 올바르게 사용하는지 런타임 검증 필요

## P1 Blockers

### B3: TensorRegistry pending/committed 분리 미구현
- **상태**: 설계서 권장이나 미구현
- **영향**: P1 — handle이 commit 전에 외부에서 관측 가능
- **미치는 영향**: concurrent graph 없으면 실질적 문제 없음 (Release 1은 직렬화)

### B4: 1,000-step Memory Test 미실행
- **상태**: 테스트 미작성/미실행
- **영향**: P1 Gate
- **필요 조건**: Browser E2E 환경

### B5: Documentation 정정 미완료
- **상태**: 감사 진행 중
- **영향**: P1 — 오해 소지 있는 문서 표현

### B6: CPU/GPU Forward Parity 미검증
- **상태**: 실제 WebGPU에서 numerical comparison 미실행
- **영향**: P1 Gate

### B7: Device Lost 시나리오 테스트 미실행
- **상태**: generation tracking은 QuotaManager에 존재하나 테스트 미실행
- **영향**: P1

## 비차단 사항 (Non-blocking)

- gpuCore.ts의 개별 op 함수들(matmul, relu 등)은 graphExecutor 경로가 아님 — allocator 전환 완료
- Python tests(36개 기존)는 CPU-only 테스트라 GPU bridge 변경에 영향받지 않음
- `sourceContract.test.ts`가 @types/node 추가 후 정상 통과
