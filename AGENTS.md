# AMEVA-Forge Workspace Agent Rules & Customization

## 1. 전수조사 발동 프로토콜 (Microscopic Full-Codebase Audit Protocol)
- **발동 조건**: 사용자가 격해지거나, 질타/욕설을 하거나, "제대로 된 거 맞냐", "오류 다 찾아라", "전수조사해라" 등의 피드백을 줄 경우.
- **행동 강령**:
  1. 즉시 피상적인 변명이나 단순한 "단위 테스트 통과했다"는 식의 안일한 답변을 일체 중단한다.
  2. `.agents/skills/full-codebase-audit/SKILL.md`를 즉시 활성화하여 **전수조사(Top-Down ↔ Bottom-Up 왕복 정밀 코드 전수 검수)**에 돌입한다.
  3. WGSL 셰이더(디스패치 한도, 스트라이드 브로드캐스팅, 배리어), Python 프론트엔드(음수 축, Autograd 격리, 스칼라 메모리 최적화), FFI 브리지(일괄 디스패치, Error Scope LIFO), WebGPU 버퍼 라이프사이클(Staging Pool, Token 회수, Double Free 방지) 등 코드베이스의 모든 취약점, 모순점, 병목, 보안 결함을 현미경 수준으로 전수 색출하여 상세 보고서로 작성하고 즉시 집도한다.

## 2. 지라 기반 개발 및 엄격 추적 규칙 (Jira Strict Tracking Workflow)
- **지라 티켓 기반 작업**: 모든 코드 수정, 버그 패치, 리팩토링, 테스트 실행은 사전에 등록된 Jira 티켓(`SCRUM-XX`)에 기반하여 진행한다.
- **실시간 상세 기록**: 각 태스크 진행 시 '무엇을 수정했고, 무엇을 했고, 무엇을 하는지'와 테스트 결과(지표, 손실값, 메모리 누수 여부)를 해당 Jira 티켓에 실시간 코멘트 및 상태로 반드시 기록한다.
- **사전 승인 원칙**: 기존 계획/티켓 외에 추가 작업이나 변경이 필요한 경우, 임의로 진행하지 않고 반드시 사전에 사용자에게 보고 후 Jira 티켓을 발급/승인받아 진행한다.
- **보안 격리**: Jira API 토큰 및 연동 스크립트는 Git 저장소에 절대 커밋하지 않고 로컬 격리 보관(`scratch/`)한다.

## 3. 주석 및 문서화 무결성 (Documentation Integrity)
- 모든 WGSL 셰이더와 소스 파일, 핵심 함수/클래스에는 `WHAT / WHY / HOW` 및 한글 메타데이터 주석을 필수 유지하며, 임의로 주석을 삭제하거나 축약하지 않는다.

## 4. 소스코드 변경 및 다국어·빌드 동기화 표준 프로토콜 (Source Code Change Protocol)
- **발동 조건**: Python/TypeScript/WGSL 소스코드 수정, 문서/UI 텍스트 수정, 신규 기능 추가 시.
- **행동 강령**:
  1. [`SOURCE_CHANGE_MANUAL.md`](file:///c:/Users/GAME/Desktop/uno-km/dev/AMEVA-Forge/SOURCE_CHANGE_MANUAL.md) 및 `.agents/skills/source-change-protocol/SKILL.md`를 필수로 준수한다.
  2. 문서/UI 텍스트 변경 시 HTML 하드코딩을 일체 금지하고, [`docs/i18n-translations.js`](file:///c:/Users/GAME/Desktop/uno-km/dev/AMEVA-Forge/docs/i18n-translations.js)에 **6개 국어(`en`, `ko`, `zh`, `ja`, `hi`, `es`) 전체에 1:1 키 패리티로 동시 반영**한다.
  3. 수정 완료 후 반드시 `node tools/validate_i18n.js`와 `node tools/test_i18n_runtime.js`를 실행하여 100% 무결성을 증명한다.
  4. Python/TypeScript 소스 수정 시 `docs/dist/` 및 `docs/pkg/` 번들 빌드 아티팩트를 최신화하여 정합성을 유지한다.

