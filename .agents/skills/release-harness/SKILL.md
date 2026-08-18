---
name: release-harness
description: >-
  AMEVA-Forge 릴리즈 2~5 개발 시 Jira 티켓 기반 실시간 작업 이력, 수정 파일, 테스트 지표,
  에러 발생 시 분석 코멘트를 자동으로 기록하고 추적하는 릴리즈 하네스 스킬.
---

# 🚀 AMEVA-Forge 릴리즈 개발 및 실시간 추적 하네스 스킬 (Release Harness Skill)

## 📌 핵심 원칙 (Core Rules)

1. **Jira 티켓 1:1 매핑**:
   - `JIRA_RELEASE_TASKS_MATRIX.md`에 등록된 `SCRUM-XXX` 티켓에 기반하여 작업을 시작하고 종료한다.
2. **실시간 수정 내역 및 메트릭 기록 (Log on Every Edit & Test)**:
   - 파일 수정 및 테스트 실행 시 `python tools/release_harness.py log <TICKET> --file <FILE> --action <ACTION> --metrics <METRICS>` 명령으로 실시간 기록한다.
3. **에러 발생 시 즉시 실패 코멘트 기록 (Fail Log on Error)**:
   - 컴파일, 셰이더 바인딩, 테스트 실패 시 `python tools/release_harness.py fail <TICKET> --error "<DETAILS>"`를 실행하여 에러와 원인을 등록한다.
4. **작업 완료 시 증빙 기록 (Mark Done on Pass)**:
   - 모든 유닛/E2E 테스트 통과 시 `python tools/release_harness.py done <TICKET> --summary "<SUMMARY>"`를 실행한다.

---

## 🛠️ CLI 사용법 (Usage Commands)

```bash
# 1. 태스크 시작
python tools/release_harness.py start SCRUM-201

# 2. 코드 수정 및 테스트 지표 기록
python tools/release_harness.py log SCRUM-201 --file "packages/forge/src/tensor/kernels/matmul_tiled.wgsl" --action "Implemented 16x16 shared memory tile" --metrics "Throughput: 4.2x faster, VRAM: 0 Leak, 100% PASS"

# 3. 에러 발생 시
python tools/release_harness.py fail SCRUM-201 --error "Workgroup barrier syntax mismatch in WGSL line 42"

# 4. 태스크 완료
python tools/release_harness.py done SCRUM-201 --summary "16x16 Tiled MatMul kernel completed and verified against PyTorch reference"

# 5. 전체 진행 상황 조회
python tools/release_harness.py status
```
