# 🚀 AMEVA-Forge Release 3.0 Official Release Notes
## ── Universal Plug & Play On-Device WebGPU AI Runtime & Model Hub ──

**Release Version**: `3.0.0` (Core Engine: `@uno-km/ameva-forge` / Python: `ameva-forge`)  
**Release Date**: 2026-09-04  
**Audit & Quality Standard**: 100% Zero-Trust Microscopic Verified (49 Test Suites, 279 Tests PASS)  
**License**: Apache-2.0  

---

### 🌟 Release Highlights (주요 핵심 성과)

> **"브라우저를 진정한 온디바이스 AI 슈퍼컴퓨터로: 1줄 삽입, WebGPU 바인딩, 외부 GGUF 모델 가중치 끼워넣기(Plug & Play)로 완성되는 제로-서버 실행기"**

1. **Plug & Play On-Device GGUF Model Hub (`SCRUM-345 ~ SCRUM-348`)**
   * Hugging Face CDN URL 또는 로컬 `.gguf` 파일 드래그 앤 드롭을 통해 외부 모델 가중치를 브라우저 WebGPU VRAM에 1-클릭으로 직결(Direct DMA).
   * 32비트 WASM 2GB 메모리 락을 우회하여 브라우저 OOM 크래시를 원천 차단.
   * 브라우저 영속 캐시(CacheStorage/OPFS) 지원으로 2회차 로드 시 네트워크 트래픽 0 및 0초 즉각 구동.
   * `SmolLM-135M-Instruct (~85MB)`, `Qwen2.5-0.5B-Instruct (~350MB)`, `LLaMA-3.2-1B-Instruct (~780MB)` 공식 추천 프리셋 카탈로그 탑재.

2. **Universal Byte-Level BPE & SentencePiece Tokenizer (`SCRUM-341`, `SCRUM-344`)**
   * GGUF 메타데이터(`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`) 파싱을 통한 어휘 사전 자동 구축.
   * LLaMA-3, SmolLM, Qwen, Gemma 계열 텍스트의 100% 가역(Invertible) 무손실 인코딩/디코딩 보장.

3. **Autoregressive Streaming Generator & Shifted Softmax Sampler (`SCRUM-342`, `SCRUM-343`)**
   * Shifted Exp 수치 안정화가 적용된 Top-K, Top-P, Temperature, Repetition Penalty 통합 샘플러.
   * 브라우저 렌더 이벤트 루프 협력적 양보(`yieldToEventLoop`)를 통한 60 FPS 논블로킹 실시간 타자기 스트리밍.

4. **Web Worker Background Neural Runner (`SCRUM-349 ~ SCRUM-352`)**
   * 무거운 트랜스포머 디코딩과 WebGPU 연산을 전용 백그라운드 Web Worker로 물리적 격리.
   * 대규모 텍스트 생성 중에도 메인 스레드 60fps UI 매끄러움 유지 및 OS GPU TDR 타임아웃 방지.

5. **Interactive Live WebGPU Studio & 6-Language i18n (`SCRUM-353 ~ SCRUM-356`)**
   * `docs/demo.html`: Hugging Face URL 입력, 드롭존, 실시간 프로그레스 바, 실시간 TPS 속도계, 대화형 챗봇 UI 제공.
   * `docs/i18n-translations.js`: 6개 국어(`en`, `ko`, `zh`, `ja`, `hi`, `es`) 전체 324개 키 100% 패리티 동기화 (`validate_i18n.js` 통과).

---

### 📦 Installation & Distribution Status

#### Python Package (PyPI)
* **Package Name**: `ameva-forge`
* **Command**: `pip install ameva-forge`
* **Distribution Status**: PyPI 공식 인덱스 배포 활성화 완료 ([pypi.org/project/ameva-forge/](https://pypi.org/project/ameva-forge/)). GitHub Release 태그 푸시 시 GitHub Actions (`publish-pypi.yml`)에 의해 자동 빌드/업로드됩니다.

#### JavaScript / TypeScript SDK (NPM / Browser)
* **Package Name**: `@uno-km/ameva-forge`
* **CDN / Browser Embed**: `<script src="https://uno-km.vercel.app/lib/forge/dist/index.js"></script>`
* **Package Distribution**: GitHub Packages & NPM 지원.

---

### 🧪 Full Test Matrix Verification

| Component | Test Suite | Tests Count | Status | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Tokenizer** | `tests/bpe_tokenizer.test.ts` | 9 | **PASS** | Round-trip 무손실 복원 |
| **Model Loader** | `tests/model_loader.test.ts` | 4 | **PASS** | GGUF 바이너리 E2E 파싱 |
| **Web Worker** | `tests/worker_session.test.ts` | 3 | **PASS** | 메시징 프로토콜 무결성 |
| **All-Modal Engine** | `tests/all_modal_orchestrator.test.ts` | 8 | **PASS** | STT, LLM, Vision, TTS GPU 연동 |
| **Core & Tiled MatMul** | `tests/matmul_tiled.test.ts` 등 45개 스위트 | 255 | **PASS** | WebGPU 셰이더 및 오토그라드 전체 통과 |
| **Total Full Suite** | **49 Test Suites** | **279 Tests** | **100% PASS** | Zero Failure / Zero Silent Fallback |
