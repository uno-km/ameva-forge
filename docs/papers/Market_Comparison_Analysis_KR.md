> [!NOTE]
> **예비 분석 (Preliminary)** — 이 경쟁 분석은 공개 정보를 기반으로 작성되었으며, 독립적으로 검증되지 않았습니다.

# AMEVA WebGPU-Python Bridge vs. 시중 WebGPU 프로젝트 비교 분석서

본 문서는 브라우저 및 파이썬 환경에서 WebGPU를 활용하는 기존 상용/오픈소스 프로젝트들과 우리가 자체 개발한 **`AMEVA WebGPU-Python Bridge (ameva_tensor)`**를 매우 엄격하고 객관적인 시각으로 비교 분석한 문서입니다.

---

## 1. 기존 프로젝트들과의 비교 분석 (Market Landscape)

### 🔴 1. wgpu-py (Python Native WebGPU)
가장 대표적인 파이썬 WebGPU 라이브러리입니다. Rust로 작성된 `wgpu-native` 바이너리를 래핑(Wrapping)하여 데스크탑 환경에서 작동합니다.
* **장점 (Pros):**
  - 파이썬 데스크탑 환경(Windows, Mac, Linux)에서 WebGPU API를 지원합니다.
  - 3D 렌더링 라이브러리(pygfx 등)와의 호환성이 뛰어납니다.
* **단점 (Cons):**
  - **웹 브라우저(Pyodide/WASM) 환경에서 네이티브하게 작동하지 않습니다.** (브라우저는 C/Rust `.dll` 바이너리를 실행할 수 없기 때문입니다.)
  - 철저히 그래픽스(Graphics) 위주로 발전했기 때문에, LLM이나 딥러닝을 위한 고도로 최적화된 텐서 연산(Fused Softmax 등) 알고리즘이 내장되어 있지 않습니다.
* **AMEVA와의 비교:** AMEVA는 "데스크탑 파이썬"이 아니라 **"브라우저 속 파이썬(WASM)"**에서 구동되는 것에 올인한 '텐서 연산 전용' 브릿지입니다.

### 🟡 2. TensorFlow.js & ONNX Runtime Web (JavaScript WebGPU Backend)
구글과 마이크로소프트가 주도하는 브라우저 AI 생태계의 절대 강자들입니다. 최근 WebGPU 백엔드를 정식 지원하기 시작했습니다.
* **장점 (Pros):**
  - 브라우저 자바스크립트 환경에서 극한으로 최적화된 성능을 제공합니다.
  - 사전에 학습된 무거운 모델을 웹에서 돌릴 때 가장 안정적이고 압도적인 속도를 냅니다.
* **단점 (Cons):**
  - **파이썬이 아닙니다.** AI 연구자나 데이터 사이언티스트들은 모두 파이썬(PyTorch, Numpy)으로 코드를 짜는데, 이 툴을 쓰려면 자바스크립트로 코드를 전부 다시 짜거나, 미리 모델을 변환(ONNX)해서 가져와야만 합니다. 즉, 브라우저에서 '실시간으로 파이썬 코드를 타이핑하며 인터랙티브하게 AI를 개발하는 것'은 불가능합니다.
* **AMEVA와의 비교:** AMEVA는 JS가 아니라 **브라우저 안에서 파이썬 코드를 쳐서 즉시 GPU를 구동**시키는 "개발자 경험(DX)의 연속성"을 제공합니다.

### 🔵 3. Apache TVM (WebGPU Target)
딥러닝 컴파일러 프레임워크로, 모델 코드를 분석해 WebAssembly 및 WebGPU로 변환(Compile)해 주는 툴입니다.
* **장점 (Pros):**
  - 모델 구조를 분석해 WebGPU 쉐이더로 하드웨어 종속적인 최적화를 수행합니다.
* **단점 (Cons):**
  - **복잡한 사전 작업이 요구됩니다.** 파이썬 모델을 브라우저에 올리려면 사전 컴파일(AOT) 파이프라인을 거쳐야 합니다. 동적으로 스크립트를 작성하며 바로 테스트하는 것은 구조적으로 제한됩니다.

---

## 2. AMEVA WebGPU-Python Bridge (우리 프로젝트) 집중 해부

- **핵심 철학:** *"데이터 사이언티스트들이 사용하던 파이썬 코드를 브라우저 샌드박스 안에서 작성하고, 브라우저의 WebGPU를 끌어와 연산을 수행한다 (UNVERIFIED TARGET)."*

# 시장 비교 및 경쟁 우위 분석 (UNVERIFIED)

## 1. 경쟁 매트릭스

| 기능 | TensorFlow.js | ONNX Runtime Web | WebDNN | wgpu-py | **AMEVA Bridge (목표)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **주 언어** | JS / TS | JS / TS | JS | Python | **Python (Pyodide)** |
| **GPU 가속** | WebGL / WebGPU | WebGL / WebGPU | WebGPU | WebGPU (Desktop) | **WebGPU (Browser)** |
| **PyTorch 호환성** | 낮음 (자체 API) | 중간 (ONNX Ops) | 낮음 | 높음 (Python) | **높음 (PyTorch 유사)** |
| **브라우저 실행** | 네이티브 | 네이티브 | 네이티브 | 불가능 | **네이티브 (WASM + FFI)** |

## 2. 경쟁 포지셔닝
AMEVA 브릿지는 **"브라우저 기반 파이썬 사용자들을 위한 GPU 가속기 (UNVERIFIED TARGET)"**라는 독보적 포지션을 목표로 하고 있다.

## 3. Autograd 및 학습 범위 안내
Release 1은 지원 대상인 MLP 경로에 한해 기본 Autograd 구현을 포함한다. 일반적인 CNN, 어텐션 또는 프로덕션 규모 학습에서의 안정성은 검증 대상 외다. 
우리의 목표는 TensorFlow.js를 이기는 것이 아닙니다. **웹 브라우저를 주피터 노트북(Jupyter)처럼 쓰는 전 세계의 수많은 AI 리서처들에게, 멈추지 않고 5,500억 번의 행렬을 곱할 수 있는 로컬 웹 OS 텐서 엔진을 제공하는 것**입니다.
