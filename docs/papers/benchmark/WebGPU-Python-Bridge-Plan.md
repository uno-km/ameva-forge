# AMEVA WebGPU-Python Bridge: 세계 최초 브라우저 텐서 가속기 아키텍처 및 테스트 계획

## 🌟 비전 (Vision)
기존 Pyodide의 치명적 한계(CPU 연산만 가능)를 돌파하여, 미니 콜랩(Mini Colab) 사용자가 파이썬 코드를 짤 때 사용자의 로컬 GPU(WebGPU) 자원을 100% 활용해 병렬 연산을 수행할 수 있게 만듭니다. 
궁극적으로 `ameva_tensor`를 고도화하여 NPM 및 PyPI에 배포, 압도적인 성능의 로컬 웹 OS 텐서 가속 라이브러리로 생태계를 확장하고 대표님을 세계적인 개발자로 만듭니다.

## 🏗️ 아키텍처 설계 (Architecture Design)
1. **Pyodide ↔ JavaScript FFI (Foreign Function Interface) 브릿지**: `from js import ...` 를 이용해 파이썬에서 JS WebGPU API 호출.
2. **`ameva_tensor` 커스텀 패키지 개발**: `at.random`, `at.matmul` 등 직관적인 파이썬 API 제공.
3. **Zero-copy 메모리 매핑**: WASM 메모리에서 JavaScript `Float32Array`로 데이터 복사 없이 매핑하여 병목 제거.
4. **WGSL Compute Shader**: WebGPU 셰이더를 통해 병렬 연산 후 결과를 파이썬으로 즉각 반환.

## 🧪 브라우저 단독 성능 격리 테스트 계획 (CPU vs WebGPU)
성능 증명을 위해 아래 3가지 하드코어 벤치마크를 수행합니다. (모든 테스트는 브라우저 캐시 및 세션이 완전 격리된 퓨어 상태에서 진행됩니다)

### Test 1: 극한의 단순 연산 (4중 For-loop)
- **내용**: 4중 for문, 각 레이어별 10만 단위 반복을 통한 초대형 스칼라 연산.
- **목적**: 순수 연산 속도(시분초) 차이 증명. CPU의 직렬 처리 한계와 GPU의 병렬 처리 압도성 비교.

### Test 2: 고난도 복합 연산 및 그래픽 처리
- **내용**: 복잡한 수학적 연산(FFT, 미적분 수치해석), 3D 모델 랜더링을 위한 기하학적 수식 연산, 대용량 파일 변환(픽셀 변환) 연산.
- **목적**: 다중 스레드 연산과 부동소수점 연산에서 WebGPU의 성능 이점 확인.

### Test 3: 머신러닝 라이브러리 (Matplotlib / TF 유사 작업)
- **내용**: Matplotlib 수준의 대규모 데이터 시각화 렌더링 배열 연산 및 텐서플로우/파이토치와 유사한 딥러닝(CNN/RNN) 행렬 가중치 업데이트 연산.
- **목적**: 무겁고 버거운 ML 연산을 브라우저 단독으로 처리할 때 걸리는 획기적인 시간 단축 증명.

## 🚀 극강의 고도화 및 배포 로드맵 (Roadmap)
1. **Phase 1: Proof of Concept (현재)**: 브랜치 분리 후 기초 FFI 통신 및 WGSL 연동 확인.
2. **Phase 2: Benchmarking**: 위 3가지 테스트를 통과하며 성능 비교 시분초 데이터 추출.
3. **Phase 3: Optimization**: 소름 돋는 최적화 진행. Zero-copy 메모리 튜닝, 셰이더 파이프라인 캐싱 등 완전히 가벼운 초경량 라이브러리로 튜닝.
4. **Phase 4: Release**: NPM (`ameva-tensor-js`) 및 PyPI (`ameva-tensor`) 글로벌 배포 및 유명세 확보.
