# 🏛️ AMEVA-Forge Master Jira Task Matrix & Real-time Tracking Board
## ── Release 1.0 (Done) ~ Release 2.0 / 3.0 / 4.0 / 5.0 Complete Catalog ──

> **규칙**: 모든 개발, 버그 수정, 셰이더 작성, 단위/E2E 테스트 실행은 아래 등록된 Jira 티켓(`SCRUM-XX`)에 1:1로 매핑되어 실시간 진행 내역과 테스트 수치(Loss, VRAM, Pass/Fail)가 기록됩니다.

---

## 🟢 [Release 1.0: Foundation Lockdown] (ALL CLOSED / DONE)

| Ticket | Task Title | Status | Result / Commit |
| :--- | :--- | :---: | :--- |
| `SCRUM-101` | 8D Stride & Tensor Handle Architecture Core | **DONE** | 112-Byte Uniform Layout 100% Contract Pass |
| `SCRUM-102` | 2D Dispatch Grid (65535x65535 Workgroup Partitioning) | **DONE** | $2^{32}$ Max Element Dispatch Guard Verified |
| `SCRUM-103` | Reverse-Mode Autograd DAG Engine & `_ctx` Isolation | **DONE** | Vector-Jacobian Backward Chain OK |
| `SCRUM-104` | Staging Buffer Recycling & AllocationToken Lifecycles | **DONE** | Zero-Leak Deterministic Pool GC Verified |
| `SCRUM-105` | Pyodide Virtual Filesystem Bundle (`forge-py-bundle.js`) | **DONE** | Offline In-Browser Python Runtime OK |
| `SCRUM-106` | In-Place Asynchronous SGD Optimizer (`axpy.wgsl`) | **DONE** | WebGPU Native In-Place Gradient Updates |
| `SCRUM-107` | Enterprise Multi-Tier i18n Engine (`docs/i18n.js`) | **DONE** | LocalStorage/IndexedDB/Cookie Persistence |
| `SCRUM-108` | 6-Language Translation Dictionary (942 Entries) | **DONE** | `en`, `ko`, `zh`, `ja`, `hi`, `es` 100% Key Parity |
| `SCRUM-109` | Smart First-Visit Geo & Locale Auto-Detection | **DONE** | Timezone / Country Mapping (India -> en default) |
| `SCRUM-110` | CI/CD Cross-Platform Shell & Unit Test Gates | **DONE** | TypeScript 104 Tests + Python 180 Tests PASS |

---

## 🚀 [Release 2.0: LLM & Transformer Acceleration Suite] (ACTIVE BRANCH: `release/v2.0`)

### Phase 2.1: Tiled MatMul with Shared Memory (`var<workgroup>`) (SCRUM-201 ~ SCRUM-208)
- [ ] `SCRUM-201`: 16x16 Workgroup Shared Memory Tile MatMul WGSL 커널 (`matmul_tiled.wgsl`)
- [ ] `SCRUM-202`: 2D Register Micro-Tiling (2x2 / 4x4 per invocation) 적용으로 글로벌 메모리 읽기 극소화
- [ ] `SCRUM-203`: GEMM In-Place Bias & Activation Fusion 커널 (`matmul_bias_relu.wgsl` 고도화)
- [ ] `SCRUM-204`: 4D Batched MatMul with Shared Memory (`[B, H, M, K] x [B, H, K, N]`)
- [ ] `SCRUM-205`: MatMul FLOPS & Bandwidth Throughput 자동 벤치마크 하네스
- [ ] `SCRUM-206`: Subgroup Matrix (Cooperative Matrix) WebGPU 실험적 확장 가드
- [ ] `SCRUM-207`: 비 16의 배수 셰이프 경계 가드 및 제로 패딩 셰이더
- [ ] `SCRUM-208`: MatMul Autograd Backward 커널 (Shared Memory Transpose-Free)

### Phase 2.2: FlashAttention-WebGPU Fused 1-Pass Kernel (SCRUM-209 ~ SCRUM-216)
- [ ] `SCRUM-209`: FlashAttention-2 Forward WGSL 커널 (Online Softmax + Running Max/Sum)
- [ ] `SCRUM-210`: Multi-Head Attention (MHA) & Grouped Query Attention (GQA) WGSL 커널
- [ ] `SCRUM-211`: Causal Masking In-Kernel Branchless Fused Attention
- [ ] `SCRUM-212`: FlashAttention Backward Pass WGSL 커널 ($O(N)$ VRAM Autograd)
- [ ] `SCRUM-213`: PyTorch `F.scaled_dot_product_attention` 1:1 FFI 바인딩
- [ ] `SCRUM-214`: SeqLen 2048/4096 VRAM Stress & PyTorch Numerical Parity Test
- [ ] `SCRUM-215`: Cross-Attention Support for Encoder-Decoder Transformer
- [ ] `SCRUM-216`: FlashAttention Softmax Scale Factor $\frac{1}{\sqrt{d_k}}$ Constant Injection

### Phase 2.3: Paged KV Caching & Generation Pipeline (SCRUM-217 ~ SCRUM-224)
- [ ] `SCRUM-217`: Paged Attention Virtual Block Table & Token Memory Manager
- [ ] `SCRUM-218`: KV Cache Slot Mapping WGSL 커널 (Prefill vs Decode 단계 분리)
- [ ] `SCRUM-219`: Rotary Position Embedding (RoPE) In-Place WGSL 커널
- [ ] `SCRUM-220`: RMSNorm (Root Mean Square Normalization) WGSL 커널
- [ ] `SCRUM-221`: SwiGLU (Swish Gated Linear Unit) Fused Activation 커널
- [ ] `SCRUM-222`: Greedy / Top-P / Top-K / Temperature Sampling WGSL 커널
- [ ] `SCRUM-223`: Web Worker Streaming Token Event Loop & UI Decoupling
- [ ] `SCRUM-224`: LLM Generation End-to-End Architecture Integration Test

### Phase 2.4: Native FP16 & Half-Precision Support (SCRUM-225 ~ SCRUM-232)
- [ ] `SCRUM-225`: WebGPU `shader-f16` Feature Detection & Adapter Negotiation
- [ ] `SCRUM-226`: WGSL `f16` Type Conversion & Software Fallback Polyfill
- [ ] `SCRUM-227`: FP16 Tiled MatMul & FP16 FlashAttention-2 Shaders
- [ ] `SCRUM-228`: FP16 Tensor Storage Buffer Allocation & 2-Byte Stride Indexing
- [ ] `SCRUM-229`: Python `forge.float16` / `forge.half` DType & NumPy FFI 연동
- [ ] `SCRUM-230`: Mixed-Precision Autograd & Loss Scaling (AMP Engine)
- [ ] `SCRUM-231`: FP16 NaN/Inf Underflow & Overflow Safe Clamping Guard
- [ ] `SCRUM-232`: VRAM 50% 절감 및 처리량 $2\times$ 벤치마크 실증

### Phase 2.5: Quantization & 4-bit/8-bit Dequant (SCRUM-233 ~ SCRUM-240)
- [ ] `SCRUM-233`: INT8 Dynamic Quantization & Scaled MatMul Shader
- [ ] `SCRUM-234`: INT4 AWQ / GPTQ Dequantization Shader (`unpack_4bit.wgsl`)
- [ ] `SCRUM-235`: GGUF Q4_K_M / Q8_0 Header & Tensor Weight Unpacker
- [ ] `SCRUM-236`: Fused Quantized MatMul (`Q4_MatMul.wgsl` On-the-fly)
- [ ] `SCRUM-237`: 1.5B ~ 7B LLM 브라우저 2GB VRAM 로드 실증
- [ ] `SCRUM-238`: PyTorch `torch.ao.quantization` 호환 Python API
- [ ] `SCRUM-239`: Quantized vs Full-Precision Perplexity Validation
- [ ] `SCRUM-240`: Release 2.0 Final Integration Lockdown & Documentation Sync

---

## 🎨 [Release 3.0: Vision & Multi-Modal Acceleration Suite] (SCRUM-301 ~ SCRUM-340)

- `SCRUM-301 ~ SCRUM-308`: Winograd $F(2 \times 2, 3 \times 3)$ Fused Conv2d & ConvTranspose2d
- `SCRUM-309 ~ SCRUM-316`: BatchNorm2d, GroupNorm, LayerNorm2d GPU State Tracking
- `SCRUM-317 ~ SCRUM-324`: Zero-Copy SafeTensors / GGUF Web Stream Parser (HTTP Range-Request)
- `SCRUM-325 ~ SCRUM-332`: ViT (Vision Transformer) Patch Embedding & Stable Diffusion VAE/UNet
- `SCRUM-333 ~ SCRUM-340`: Multi-Modal Audio WebCodecs Bridge & Whisper WebGPU Real-time STT

---

## 🌐 [Release 4.0: Distributed Edge & WebRTC Federated Learning] (SCRUM-401 ~ SCRUM-440)

- `SCRUM-401 ~ SCRUM-408`: WebRTC DataChannel Zero-Copy Binary Tensor Chunk Streamer
- `SCRUM-409 ~ SCRUM-416`: WebRTC Ring-AllReduce & P2P Mesh Gradient Sync Protocol
- `SCRUM-417 ~ SCRUM-424`: Federated Learning Coordinator & Client Aggregator Protocol
- `SCRUM-425 ~ SCRUM-432`: On-Device LoRA (Low-Rank Adaptation) Real-time Fine-Tuning Engine
- `SCRUM-433 ~ SCRUM-440`: Differential Privacy Gradient Clipping & Edge Privacy Guarantees

---

## ⚡ [Release 5.0: Compiler Graph JIT & Enterprise Ecosystem] (SCRUM-501 ~ SCRUM-540)

- `SCRUM-501 ~ SCRUM-508`: WebGPU Kernel Fusion JIT Compiler (Elementwise Chaining)
- `SCRUM-509 ~ SCRUM-516`: ONNX / TorchScript Intermediate Representation (IR) Transpiler
- `SCRUM-517 ~ SCRUM-524`: WebCodecs Hardware Video/Audio Streaming Pipeline
- `SCRUM-525 ~ SCRUM-532`: Multi-GPU / Web Worker Pipeline Parallelism
- `SCRUM-533 ~ SCRUM-540`: Production Global NPM Distribution, CDN Artifacts & LTS Lockdown
