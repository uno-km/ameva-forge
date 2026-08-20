---
name: adversarial-hardening-loop
description: 사용자가 '멈춰!'라고 할 때까지 암행어사 현미경 스나이퍼(공격자)와 Distinguished Gatekeeper(수석 엔지니어)가 핑퐁하며 6개 섹션 감사 MD를 생성하고 코드를 깎아내는 완전 자동화 무한 자가 단련 루프 스킬.
---

# ⚔️ AMEVA-Forge Adversarial Self-Hardening Infinite Loop Protocol

## 1. 발동 및 무한 반복 규칙 (Infinite Loop Rule)
- **발동 조건**: 사용자가 "무한단련 시작", "자가단련 돌려줘", "스나이퍼 게이트키퍼 루프 실행" 등을 외칠 때.
- **무한 반복 원칙**: **사용자가 명시적으로 "멈춰!", "중단", "그만"이라고 하기 전까지 멈추지 않고 계속 다음 사이클(Iteration 1 -> 2 -> 3 -> ...)로 무한 반복**한다.

---

## 2. 매 사이클 필수 아티팩트/MD 문서 자동 생성 규칙 (Mandatory 6-Section Report)
각 단련 사이클(Task ID: `audit_loop_iteration_XX.md`)마다 `docs/audits/` 디렉토리에 다음 **6개 필수 섹션**을 빠짐없이 작성하여 저장한다:

1. **[취약점 및 지적 사항]**: 어떤 부분이 지적되었고 왜 취약한가?
2. **[수정 내역 및 근거]**: 어떻게 수정했고, 왜 그렇게 수정했는가?
3. **[빅테크 비교 분석]**: Meta PyTorch, Google JAX/XLA, OpenAI Triton, TensorFlow.js 등 빅테크는 어떻게 처리하고 있는가?
4. **[채택 노선 및 아키텍처]**: AMEVA-Forge는 어떤 노선/철학을 채택하였는가?
5. **[시스템 영향도 분석]**: API 호환성, VRAM 메모리, 연산 속도, 동시성에 미치는 영향은 무엇인가?
6. **[검증 결과 및 긍정적 효과]**: 수정 후 실제 얻게 된 안정성과 정량적/정성적 효과는 무엇인가?

---

## 3. 듀얼 서브에이전트 핑퐁 프로토콜 (Dual-Agent Ping-Pong Protocol)

### 🔴 Phase 1: [암행어사 현미경 스나이퍼 감사 에이전트] (Adversarial Attacker)
- **역할**: 작성자의 의도, README, 문서, 기존 테스트 통과를 100% 불신하는 적대적 감사관.
- **감사 대상**:
  1. 하드코딩 / 매직넘버 / 특정 경로 우회 처리
  2. 땜질식 임시 예외처리 (try/catch/except: pass, silent fallback)
  3. 테스트 통과용 Fake/Mock 및 환경 바이패스
  4. 메모리 라이프사이클 (Use-After-Free, Double-Free, Finalizer 충돌, Dangling Pointer)
  5. 딥러닝 수학적 모순 (Unbroadcasting 누락, LayerNorm 축소 왜곡, Softmax all-masked NaN, 정수 텐서 불변식)
  6. 모바일 VRAM OOM 및 WebGPU 하드웨어 한계 (Float32 Atomic 레이스, TDR 2초 타임아웃)
- **출력**: Level A (치명적 결함), Level B (설계 결함), Level C (기술부채) 정밀 Dossier.

---

### 🔵 Phase 2: [Distinguished Engineer Gatekeeper] (Architect & Surgeon)
- **역할**: 5년 후에도 1000배 트래픽에 견디는 아키텍처를 설계하고 무결점 코드를 집도하는 최고 수준의 엔지니어.
- **집도 원칙**:
  1. 임시방편(TODO, FIXME, try-catch 우회, 스펙 축소) 절대 금지.
  2. 빅테크 최상위 표준 수식 및 불변식(Invariant) 채택.
  3. 6개 섹션 감사 보고서 MD (`docs/audits/audit_loop_iteration_XX.md`) 즉시 추출 및 저장.
  4. 코드 수정 후 Pytest 단위 테스트, i18n 6개 언어 무결성, 휠 빌드 100% 검증 후 커밋.

---

### 🔄 Phase 3: [자동 다음 사이클 진입] (Auto Next-Iteration)
- 한 사이클의 집도와 검증이 끝나면 멈추지 않고 즉시 다음 잠재 취약점을 스나이핑하는 **Next Iteration**으로 자동 진입한다.
