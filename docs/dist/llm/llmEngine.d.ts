/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-334 & SCRUM-335 High-Performance On-Device LLM & BitNet 1.58b Execution Engine
 *
 * WHAT: RoPE, RMSNorm, SwiGLU, KV-Cache 및 BitNet 1.58b 3진(-1, 0, +1) 양자화를 지원하고,
 *      WebGPU WGSL FlashAttention / Tiled Matmul / SwiGLU 셰이더 기반 하드웨어 가속을 직결한 트랜스포머 디코더 엔진입니다.
 * WHY: 외부 클라우드 통신 없는 100% 로컬 텍스트 생성, 추론, 및 올모달 멀티모달 두뇌 역할을 초고속으로 수행하기 위함입니다.
 * HOW: Token Embedding -> N-Layer Decoder(RMSNorm -> QKV Proj -> RoPE -> Causal Attn -> RMSNorm -> SwiGLU MLP) -> LM Head -> Sampler.
 */
export declare enum LLMErrorCode {
    LLM_EMPTY_PROMPT = "LLM_EMPTY_PROMPT",
    LLM_WEIGHTS_REQUIRED = "LLM_WEIGHTS_REQUIRED",
    LLM_NON_FINITE_LOGITS = "LLM_NON_FINITE_LOGITS",
    LLM_CONTEXT_OVERFLOW = "LLM_CONTEXT_OVERFLOW",
    WEBGPU_NOT_AVAILABLE = "WEBGPU_NOT_AVAILABLE"
}
export declare class LLMError extends Error {
    readonly code: LLMErrorCode;
    constructor(code: LLMErrorCode, message: string);
}
export interface LLMDecoderLayerWeights {
    inputNormGamma: Float32Array;
    qWeight: Float32Array;
    kWeight: Float32Array;
    vWeight: Float32Array;
    outWeight: Float32Array;
    postNormGamma: Float32Array;
    gateWeight: Float32Array;
    upWeight: Float32Array;
    downWeight: Float32Array;
}
export interface LLMWeights {
    tokenEmbedding: Float32Array;
    layers: LLMDecoderLayerWeights[];
    finalNormGamma: Float32Array;
    lmHeadWeight: Float32Array;
}
export interface KVCache {
    k: Float32Array;
    v: Float32Array;
    length: number;
}
export declare class LLMEngine {
    static readonly DIM = 512;
    static readonly NUM_HEADS = 8;
    static readonly HEAD_DIM = 64;
    static readonly HIDDEN_DIM = 1024;
    static readonly MAX_SEQ_LEN = 512;
    /**
     * RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
     */
    static rmsNorm(x: Float32Array, gamma: Float32Array, dim: number, eps?: number): Float32Array;
    /**
     * RoPE (Rotary Position Embedding): 반차원 회전 인코딩
     */
    static applyRoPE(x: Float32Array, pos: number, dim: number, headDim: number): Float32Array;
    /**
     * SwiGLU Fused Activation: (x W_gate * silu(x W_gate)) * (x W_up)
     */
    static swiglu(x: Float32Array, dim: number, hiddenDim: number, wGate: Float32Array, wUp: Float32Array, wDown: Float32Array): Float32Array;
    /**
     * 단일 토큰 순전파 및 다음 토큰 확률 분포(Logits) 예측 (CPU Reference)
     */
    static forwardToken(tokenId: number, pos: number, weights: LLMWeights, kvCaches: KVCache[], dim?: number, vocabSize?: number): Float32Array;
    /**
     * 단일 토큰 순전파 및 다음 토큰 확률 분포(Logits) 예측 (WebGPU Hardware Accelerated):
     * WebGPU Tiled GEMM 및 SwiGLU 셰이더를 통해 VRAM 내에서 하드웨어 가속 실행합니다.
     */
    static forwardTokenGPU(tokenId: number, pos: number, weights: LLMWeights, kvCaches: KVCache[], dim?: number, vocabSize?: number): Promise<Float32Array>;
    /**
     * 토큰 시퀀스 전체 순전파
     */
    static forward(tokens: number[], weights: LLMWeights, dim?: number, vocabSize?: number): {
        logits: Float32Array;
        kvCaches: KVCache[];
    };
    /**
     * 토큰 시퀀스 전체 WebGPU 하드웨어 가속 순전파
     */
    static forwardGPU(tokens: number[], weights: LLMWeights, dim?: number, vocabSize?: number): Promise<{
        logits: Float32Array;
        kvCaches: KVCache[];
    }>;
}
