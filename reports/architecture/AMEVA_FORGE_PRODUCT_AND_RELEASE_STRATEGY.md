---
Revision: Red Team Review 2
Investigation date: 2026-08-13
Repository commit hash: N/A (Local workspace)
Test environment: Node (Jest), Native Windows Python 3.12
WebGPU E2E status: Not Run / Unavailable in Node/Native
Confidence level: High (Based on thorough static analysis and targeted local test executions)
---

# AMEVA-Forge 제품 목적, 기술 차별성 및 1차 배포 전략 보고서

## 변경 이력 (Red Team Review 2)
*   **Maturity 하향 조정:** 브라우저 GPU E2E 테스트 부재 및 `reshape` 결함 확인으로 인하여 "Production-ready" 판정을 일괄 삭제하고 "Verified Beta" 또는 "Broken"으로 하향 조정.
*   **신규 Blocker 추가:** `sum/max` reduction 로직 결함, GPU Error propagation 비동기 유실 문제, Quota ledger 붕괴, 전역객체(`globalThis`)의 raw device 노출 이슈 추가.
*   **Release 1 범위 축소:** 불안정한 CNN(`Conv2d`, `pooling`), `permute` 등 고수준 모듈을 배제하고 2-layer MLP 훈련이 가능한 최소 집합으로 축소.
*   **경쟁 분석 수정:** "유일한 대안" 등 근거 없는 서술을 배제하고 잠재적 차별성(Potential)으로 하향 평가.
*   **테스트 인프라 해석 수정:** Pytest 실패 원인이 Windows 환경 버그가 아닌 프로젝트 내부의 `sys.stdout` 래핑 버그임을 입증.

---

## 1. Executive Summary (요약)

AMEVA-Forge는 브라우저 기반 Python/WebGPU training을 지향하는 연구 및 교육용 runtime 후보입니다. 

*   **최종 판정:** **현재는 Internal Alpha 상태이며 Public Release는 불가능합니다.** MLP 수준의 제한된 Public Technical Preview를 목표로 runtime integrity와 CPU/GPU parity를 먼저 확립해야 합니다.
*   **가장 큰 위험 (P0 Blockers):** 
    1. 텐서 형태 변환(`reshape`) 호출 시 GPU 실행이 충돌(`AMEVAForgeSecurityError`).
    2. `sum`, `max` 축소 연산 시 다단계 reduction이 생략되어 부분(Partial) 결괏값만 반환하는 심각한 수학적 오류 발생.
    3. GPU Validation Error가 Python 레이어로 전파되지 않고 침묵(Swallowing)하여 실패한 그래프 연산이 성공한 것처럼 처리됨.
    4. Python 및 TypeScript 양쪽의 단위 테스트 인프라 붕괴.
*   **추천 특화 방향:** 모든 딥러닝 기능을 모방하는 범용 프레임워크가 아닌, **브라우저 내 교육용 Autograd 프레임워크 및 소형 모델 실험 플랫폼(Browser-local educational autograd)**으로 철저히 범위를 좁혀야 합니다.

---

## 2. 조사 범위 및 방법

*   **조사 패키지:** `forge-py/` (Python), `forge/` (TypeScript)
*   **실행한 명령:** `npm run test` 및 Python 부분 인라인 검증.
*   **실행하지 못한 범위:** `pyodide` 런타임 의존성으로 인하여 네이티브 Python 환경에서의 `unittest`는 전체 Skip 되었습니다. 브라우저 WebGPU E2E 테스트(Puppeteer/Playwright 등)가 부재하여 **Not Run**으로 판정했습니다.
*   **판단 기준:** 문서나 WGSL 파일의 단순 존재 여부가 아니라, Python Lazy Node 생성 -> Graph JSON 직렬화 -> TS 파싱(`ALLOWED_OPS`) -> Dispatch -> Readback으로 이어지는 E2E 경로의 무결성을 최우선 기준으로 삼았습니다.

---

## 3. 시스템 아키텍처

1.  **Python API & Autograd (`forge-py`)**: 사용자는 `fg.tensor` 등을 통해 즉시 실행(Eager)처럼 보이는 코드를 작성하나, 내부적으로는 `_lazy_op`와 `_parents`를 가지는 AST(추상 구문 트리)를 구성합니다.
2.  **Pyodide Bridge & JSON Serialization**: `.realize()` 시, 그래프를 위상 정렬한 뒤 단일 JSON 배열로 변환하여 TypeScript로 전달합니다.
3.  **TypeScript Runtime (`forge`)**: `graphExecutor.ts`가 JSON을 파싱하고, `ALLOWED_OPS`를 검사 후 WebGPU Compute Pass로 전송합니다.
4.  **Transaction의 부재**: 현재 아키텍처는 GPU Submission 중 에러가 발생해도 이전 메모리 할당(Quota)을 롤백(Rollback)하지 않고 유효하지 않은 핸들을 반환하는 치명적 결함을 가집니다.

---

## 4. 프로젝트 개발 목적

*   **대상 사용자:** 인프라 설치 없이 브라우저 단에서 순수 Python만으로 간단한 신경망을 구축하고 학습시키고자 하는 교육자 및 연구자.
*   **현실적으로 가능한 목적:** TensorFlow.js나 PyTorch 자체를 대체하려는 시도는 FFI(Foreign Function Interface) 직렬화 오버헤드와 생태계 차이로 인해 불가능에 가깝습니다. **"브라우저에서 서버 없이 구동되는 빠르고 가벼운 딥러닝 교육용 도구"**로 목적을 명확히 해야 합니다.

---

## 5. 공개 API 성숙도 전면 재평가

GPU E2E 테스트가 완전히 자동화되어 있지 않으므로 어떠한 기능도 "Production-ready"로 볼 수 없습니다.

| 기능 범주 | 연산자 | 상태 (Maturity) | 사유 (Evidence) | Release 1 포함 여부 |
| :--- | :--- | :--- | :--- | :--- |
| **Tensor Creation** | `tensor`, `upload` | Verified Beta | `upload` op 정상 매핑 및 TS 지원 확인. | O |
| **Basic Math** | `add`, `sub`, `mul`, `div`, `neg` | Verified Beta | WGSL 커널 및 TS `ALLOWED_OPS` 지원. | O |
| **Matrix Ops** | `matmul` (2D) | Verified Beta | X/Y Swap 등 디스패치 구조 존재. | O |
| **Shape Ops** | `reshape`, `unsqueeze`, `squeeze` | **Broken** | `ops.py`가 `op='reshape'`을 생성하나 `graphExecutor.ts`의 `ALLOWED_OPS`에 없어 GPU Crash 발생. | O (수정 필수) |
| **Reductions** | `sum`, `max` (Scalar) | **Broken** | TS의 Dispatch 로직 버그로 다단계 Reduction이 실행되지 않아 잘못된 결과 반환. | O (수정 필수) |
| **Activations** | `relu`, `sigmoid`, `tanh` | Verified Beta | Forward/Backward 구현 확인. | O |
| **Advanced NN** | `Conv2d`, `maxpool2d`, `im2col` | Not Verified | 패딩, 스트라이드, E2E 검증 부족. | **X** |
| **High-level NN**| `RNN`, `LSTM`, `Transformer` | Experimental | 고수준 Python 구현체일 뿐, E2E 성능/검증 부재. | **X** |

---

## 6. 경쟁 분석

| 비교 대상 | Browser WebGPU Training? | Python First? | AMEVA-Forge와의 비교 (Potential) |
| :--- | :--- | :--- | :--- |
| **PyTorch** | No (WASM 존재하나 WebGPU 훈련 공식 지원 미비) | Yes | PyTorch는 서버/네이티브 표준임. AMEVA는 PyTorch API를 모방함. |
| **TensorFlow.js** | Yes (WebGL / WebGPU 기반) | No (JS/TS 전용) | TF.js가 훨씬 안정적이나, AMEVA는 Python 생태계를 활용할 수 있다는 잠재력 보유. |
| **ONNX Runtime Web**| No (Inference 전용) | No | 훈련(Training, Autograd)이 불가능함. |
| **tinygrad** | WebGPU backend (실험적) | Yes | tinygrad는 범용을 노리나 브라우저 E2E 중심은 아님. |
| **AMEVA-Forge** | Yes (자체 WGSL & Autograd) | Yes | **브라우저 환경에서 Python + WebGPU 훈련을 지향하는 희소한 조합 (잠재적 차별성).** |

---

## 7. 차별성 평가

*   **Potential Differentiation (잠재적 차별성):** "브라우저 기반 Python WebGPU 훈련 런타임". 수학적 정확성과 안정성이 입증될 경우 매우 강력한 교육/실험용 플랫폼이 될 수 있습니다.
*   **Superficial Differentiation (표면적 차별성):** "PyTorch와 유사한 API".
*   **Disadvantage (약점):** Python -> TS로 넘어갈 때 발생하는 거대한 JSON 직렬화 병목(FFI Overhead).

---

## 8. 추천 특화 방향

*   **최종 선택:** **Browser-local educational autograd & Small model experimentation**
*   **이유:** 현재 엔진 아키텍처와 성능 격차를 고려할 때 대형 모델 상용화는 무리입니다. 3초 만에 켜지는 브라우저 딥러닝 플레이그라운드, 데이터 프라이버시가 완벽히 보장되는 로컬 소형 신경망 훈련 엔진으로 승부하는 것이 가장 타당합니다.

---

## 9. 핵심 기술 격차 (P0, P1)

| 우선순위 | 영역 | 문제 / 격차 | 필요한 작업 | 완료 기준 |
| :--- | :--- | :--- | :--- | :--- |
| **P0** | **Runtime Integrity** | `reshape` 호출 시 TS 검증 실패로 Crash 발생. | TS 런타임에서 Metadata-only View 변환 지원 혹은 커널 복사 추가. | `reshape` 후 GPU E2E 정상 출력. |
| **P0** | **Math Correctness** | `sum`과 `max` 연산이 `byteLength` 초기화 버그로 부분(Partial) 축소만 수행. | 입력 텐서의 형태를 기반으로 `currentSize` 초기화 및 다단계 루프 수정. | 65536개 이상 요소 reduction 시 CPU 결과와 동일. |
| **P0** | **Error Propagation** | `device.popErrorScope()`의 에러를 Await하지 않아 가짜 성공 반환. | `executeGraph`를 트랜잭션화하여 Validation 실패 시 핸들 발급 롤백 및 Python 예외 전파. | GPU 에러가 Python의 `AMEVAForgeDeviceError`로 발생. |
| **P1** | **Memory Ledger** | Quota 초과 시 Clamp 됨(예외 미발생). 임시 버퍼들(Reduction 등)이 추적되지 않음. | Token 기반 Quota 시스템으로 재작성. | Quota 100% 추적 및 초과 시 예외 발생. |

---

## 10. Release 1 범위 (축소 조정)

기존 "모든 기능을 담겠다"는 접근을 버리고 **"2-layer MLP 훈련의 완벽한 달성"**으로 축소합니다.

*   **반드시 포함 (Core):** `tensor`, `upload`, `add`, `sub`, `mul`, `div`, `neg`, `matmul` (2D), `relu`, `reshape`(픽스 후), `transpose`(2D), `sum`(픽스 후), SGD Optimizer.
*   **Experimental로 포함:** `exp`, `log`, `sigmoid`, `tanh`, `unsqueeze`, `squeeze`, `permute`, `dropout`.
*   **Release 1에서 제외:** `Conv2d`, `maxpool2d`, `avgpool2d`, `im2col`, `col2im`, `BatchNorm2d`, `RNN`, `Transformer` 등 검증 부하가 크고 E2E 훈련 증명이 부족한 모든 고급 모듈.

---

## 11. Release Roadmap

1.  **Internal Alpha (현재):** Runtime Integrity 붕괴 상태. 기반 공사(Blocker 제거) 집중.
2.  **Public Technical Preview (Release 1):** MLP 훈련이 정확성(Numerical Parity) 오차 없이 브라우저에서 동작하는 최초의 공개 버전.
3.  **Beta (Release 2):** CNN 모듈 지원, 브라우저 환경 CI/CD 테스트 자동화 구축.
4.  **Stable 1.0 (Release 3):** 메모리 누수 완벽 차단, JSON 오버헤드 50% 이상 개선.

---

## 12. Release Blockers (Top 6)

1.  **ID-01: `reshape` Graph Contract (Critical)**: `ops.py`는 `reshape`을 전송하나 `ALLOWED_OPS`가 차단하여 즉시 Crash 발생.
2.  **ID-02: `sum/max` Reduction Correctness (Critical)**: `graphExecutor.ts`(L512) 버그로 다단계 축소가 무시되어 엉뚱한 결괏값을 반환.
3.  **ID-03: GPU Validation Error Propagation (High)**: `device.popErrorScope()` 비동기 유실로 인해, 실패한 연산이 성공한 것으로 둔갑함.
4.  **ID-04: Quota Ledger Bypass (High)**: 임시 버퍼가 `QuotaManager`를 우회하고, 과도한 해제(duplicate release) 시 타 자원의 메모리를 잠식하여 우회 가능함.
5.  **ID-05: `globalThis` raw GPUDevice Exposure (Medium)**: JS 런타임 내 제3자 스크립트가 `__AMEVA_DEVICE__`에 접근하여 디바이스를 강제 점유/파괴 가능.
6.  **ID-06: Python Test Harness Wrapped stdout (High)**: `test_run_all.py`에서 `sys.stdout`을 강제 래핑하여 Windows 환경의 `pytest`를 근본적으로 파괴함.

---

## 13. 테스트 및 품질 전략

*   **현재 테스트 오류 원인 격리:** Pytest 실패는 Windows 버그가 아니라, 프로젝트의 `test_run_all.py`가 `sys.stdout = io.TextIOWrapper(...)`로 I/O 스트림을 덮어씀으로써 `pytest`의 `capture.py`와 충돌하는 **코드 레벨 결함**입니다.
*   **필수 과제:** Pyodide 런타임 의존성으로 인해 네이티브 환경에서 GPU 검증이 불가능합니다. Puppeteer나 Playwright를 사용한 **Headless Browser WebGPU E2E 테스트 프레임워크 구축**이 최우선입니다.
*   **CPU / GPU Parity:** CPU 연산(NumPy) 결과는 단지 Reference Path일 뿐만 아니라, GPU 로직의 정확성을 검증하는 유일한 잣대이므로 반드시 두 결과값을 비교(`np.allclose`)하는 Numerical Gradient Checks가 포함되어야 합니다.

---

## 14. 성능 전략

*   성능 우위를 증명할 신뢰성 있는 Benchmark가 없으므로 "속도가 빠르다"는 주장은 당분간 배제해야 합니다.
*   가장 큰 병목인 "Python -> TS 간 JSON 텍스트 파싱 오버헤드"를 해결하기 전까지는, Graph Validation(의미론적 검증) 등 안정성 강화에 리소스를 집중해야 합니다.

---

## 15. 최종 권고안

"AMEVA-Forge는 브라우저 기반 Python/WebGPU training을 지향하는 연구 및 교육용 runtime 후보입니다. 현재는 Internal Alpha이며, MLP 수준의 제한된 Public Technical Preview를 목표로 runtime integrity와 CPU/GPU parity를 먼저 확립해야 합니다."

---

## 16. 30일, 60일, 90일 실행 계획

*   **0~30일 (Runtime Integrity):** `reshape` 의미론 수정, `sum/max` 버그 수정, GPU 에러 전파 트랜잭션 확립, `sys.stdout` 래핑 제거 및 테스트 하네스 복구.
*   **31~60일 (Core Parity):** CPU/GPU Forward & Backward Parity 달성, Shape 퍼징(Fuzzing) 테스트, Puppeteer 기반 브라우저 WebGPU CI 구축.
*   **61~90일 (Technical Preview):** 2-layer MLP 전용 공식 예제 배포, 지원되지 않는 API 노출 정리, PyPI 패키지 메타데이터 갱신 및 Release 1 런칭.
