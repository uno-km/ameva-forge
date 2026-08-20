---
name: adversarial-hardening-loop
description: 암행어사 현미경 스나이퍼(공격자)와 Distinguished Engineer Gatekeeper(수석 엔지니어)가 서로 공격하고 방어하며 5회 이상 연속 무결점을 달성할 때까지 코드를 자동으로 깎아내는 완전 자동화 자가 단련 루프 스킬.
---

# ⚔️ AMEVA-Forge Adversarial Self-Hardening Loop Protocol

## 1. 발동 트리거 (Activation Triggers)
사용자가 다음과 같은 명령을 내릴 때 즉시 활성화된다:
- '무한단련', '자가단련', '자동으로 깎아줘', '스나이퍼 게이트키퍼 루프', '완전자동화 단련'

---

## 2. 듀얼 서브에이전트 핑퐁 프로토콜 (Dual-Agent Ping-Pong Protocol)

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
  2. 빅테크(Meta PyTorch, Google JAX/XLA, OpenAI Triton, Chrome Dawn) 최상위 표준 수식 채택.
  3. 불변식(Invariant) 기반 버그 클래스(Bug Class) 원천 박멸.
  4. 코드 수정 후 Pytest 단위 테스트, i18n 6개 언어 무결성, 휠 빌드 100% 검증.

---

### 🔄 Phase 3: [재귀적 2차 역검증 & 종료 조건] (Recursive Verification & Exit Criteria)
1. 수정된 코드를 다시 **Phase 1(공격자)**에게 전달하여 새로운 사이드 이펙트나 숨겨진 취약점이 있는지 공격한다.
2. **연속 3회 이상 Phase 1에서 Level A/B 결함이 0건으로 검출**되고, 모든 스트레스 퍼징 테스트가 통과할 때 비로소 APPROVED 판정을 내리고 Git 커밋 및 배포를 완료한다.
