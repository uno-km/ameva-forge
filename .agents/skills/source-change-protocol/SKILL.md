---
name: source-change-protocol
description: >-
  AMEVA-Forge 소스코드, 셰이더, 다국어 문서(i18n), 빌드 번들 변경 시 정합성과 무결성을 보장하는 표준 대처 프로토콜 스킬.
  모든 코드/문서 변경 시 6개 언어 동기화, 번들 갱신, 메모리 쿼터 검증, 전수조사 검증을 체계적으로 수행한다.
---

# AMEVA-Forge 소스코드 변경 및 다국어·빌드 무결성 프로토콜 (Source Code Change Protocol)

## 📌 핵심 원칙 (Core Rules)

1. **i18n 6개 국어 동시 동기화 의무**:
   - 문서(`docs/*.html`)에 텍스트를 추가/수정할 경우 하드코딩을 절대 금지하며 `data-i18n` 또는 `data-i18n-html` 속성을 부여한다.
   - [`docs/i18n-translations.js`](file:///c:/Users/GAME/Desktop/uno-km/dev/AMEVA-Tensor/docs/i18n-translations.js) 내 6개 언어(`en`, `ko`, `zh`, `ja`, `hi`, `es`) 전체에 1:1 키 패리티를 유지하여 번역 딕셔너리를 작성한다.
   - 변경 후 반드시 `node tools/validate_i18n.js`를 실행하여 100% 무결성을 증명한다.

2. **빌드 아티팩트 및 가상 파일시스템 동기화**:
   - Python 소스(`packages/forge-py/forge/`) 변경 시: 가상 파일시스템 번들(`docs/dist/forge-py-bundle.js`) 및 Wheel(`docs/pkg/*.whl`) 동기화.
   - TypeScript/WGSL 소스(`packages/forge/src/`) 변경 시: 브라우저 배포 번들(`docs/dist/index.js`) 동기화.

3. **지라 티켓 및 주석 무결성 준수**:
   - Jira 티켓(`SCRUM-XX`)에 기반하여 작업하고, 실시간 수정 내역과 테스트 수치를 기록한다.
   - 모든 파일의 `WHAT / WHY / HOW` 및 한글 메타데이터 주석을 보존한다.

---

## 🛠️ 실행 및 검증 절차 (Execution Workflow)

```bash
# 1. i18n 번역 키 패리티 및 HTML 바인딩 전수 검증
node tools/validate_i18n.js

# 2. i18n 런타임 스토리지(LocalStorage/IndexedDB/Cookie) 시뮬레이션 검증
node tools/test_i18n_runtime.js

# 3. Python 단위 테스트 검증
python -m unittest discover packages/forge-py/tests/
```

상세 내용은 워크스페이스 루트의 [`SOURCE_CHANGE_MANUAL.md`](file:///c:/Users/GAME/Desktop/uno-km/dev/AMEVA-Tensor/SOURCE_CHANGE_MANUAL.md)를 상시 참조한다.
