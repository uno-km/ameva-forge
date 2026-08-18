---
name: full-codebase-audit
description: >-
  사용자가 격해지거나, 질타/비판을 하거나, 잠재 결함·오류·모순점 점검을 요청할 때 즉시 발동하는
  전수조사(Microscopic Full-Codebase Audit) 프로토콜 스킬.
  안일한 답변이나 피상적 테스트 통과 확인을 멈추고, 코드베이스의 모든 라인, WGSL 커널, FFI 브리지,
  수학적 공식, 메모리 라이프사이클, 보안 취약점을 현미경 수준으로 전수 정밀 감사하여 보고한다.
---

# 🔬 전수조사 및 정밀 결함 감사 프로토콜 (Microscopic Full-Codebase Audit Skill)

## 📌 1. 발동 조건 (Trigger Conditions)

다음 상황 중 하나라도 발생하면 즉시 이 스킬을 최우선으로 활성화합니다:
1. **사용자의 강력한 질타, 비판, 감정적 피드백(욕설 등)**:
   - "왜 오류가 나냐", "너 한계냐", "제대로 된 거 맞냐", "다시 찾아봐" 등의 피드백이 들어오는 즉시 안일한 변명을 멈추고 전수조사에 돌입한다.
2. **"전수조사", "보안 감사", "코드 검수", "취약점 점검" 명시적 요청**:
3. **새로운 릴리즈 전 최종 락다운(Release Lockdown) 검증 시점**:

---

## 🎯 2. 전수조사 6대 감사 프레임워크 (Audit Framework)

전수조사 발동 시, 아래 6대 영역을 **Top-Down ↔ Bottom-Up으로 최소 5회 이상 교차 검증**하여 잠재 결함을 색출합니다:

### ① WebGPU 커널 및 셰이더 무결성 (Kernel & Shader Audit)
- 1D/2D 디스패치 상한(65,535 워크그룹) 초과 방어 여부
- 다차원(3D, 4D) 스트라이드 디코더 및 브로드캐스팅 좌표 역산의 수학적 정합성
- 공유 메모리(`var<workgroup>`) 동기화 배리어(`workgroupBarrier`) 누락 여부
- IEEE 754 NaN/Inf 가드 및 수치 안정성 (Division by Zero, Clamp)

### ② Python 프론트엔드 및 수식 일치성 (Python Math & Shape Audit)
- 음수 인덱스/축(`axis < 0`) 처리 시 양수 랭크 정규화 여부
- Reverse-mode DAG 및 Autograd `_ctx` 하이재킹 격리 (`with no_grad():`)
- 스칼라 연산 시 불필요한 VRAM 중복 할당 여부 (0-dim 스칼라 텐서 및 0-stride 활용)

### ③ WebGPU 메모리 라이프사이클 및 쿼터 (Memory & Quota Audit)
- Staging Buffer Pool 폐기/회수 시 `AllocationToken` 소유권 및 `freeBuffer` 일치성 (Double Free 및 쿼터 누수 차단)
- Device Lost 후 좀비 버퍼 소각(`clearStagingPool()`) 및 런타임 클린업 순서
- 맵핑 실패 또는 GPU 예외 발생 시 오염된 버퍼 풀링 차단 (`isCorrupted`)

### ④ FFI 브리지 및 비동기 동기화 (FFI Bridge & Async Concurrency)
- 옵티마이저 등 반복 연산 시 파라미터별 개별 FFI 분할 루프 병목 제거 $\to$ 단일 일괄 FFI 디스패치
- Error Scope LIFO 스택 순차 배수 (`popErrorScope` 순서 보장)
- WASM 메모리 뷰 바운드 검증 (`assertWasmRange`)

### ⑤ 보안 방어선 및 입력 검증 (Security Boundaries)
- JSON Prototype Pollution (`__proto__`, `constructor`) 차단
- 악성 op 이름 및 스키마 드리프트 방어 (화이트리스트 대조)
- In-place (`axpy`) 연산 후속 트랜잭션 오염 차단

### ⑥ 테스트 커버리지 및 회귀 방지 (Test Coverage Verification)
- 해피 패스(기본 시나리오)뿐만 아니라 엣지 케이스, 경계값, 실패 복구 시나리오 전수 검증
- Jest 단위 테스트, Python 단위 테스트, Playwright WebGPU 실제 브라우저 E2E 3계층 통합 실증

---

## 📋 3. 보고서 표준 양식 (Audit Deliverable)

전수조사 완료 시, 반드시 다음 형식으로 보고서를 생성/출력합니다:

1. **Executive Summary**: 총 분석 라인 수, 발견된 Critical/High/Medium/Low 결함 요약
2. **세부 취약점 분석 (Detail Findings)**:
   - 위치 및 코드 라인
   - 근본 원인 (Root Cause)
   - 악용/실패 시나리오 (Failure Scenario)
   - 명확한 수정 코드 및 해결책 (Code Solution)
3. **조치 및 테스트 검증 결과**:
   - TypeScript Build / Jest / Python / WebGPU E2E 수치 보고
