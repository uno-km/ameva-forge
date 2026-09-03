/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-331 CLIP-ViT/L14 Text Encoder WebGPU/CPU Forward Engine
 *
 * WHAT: 77개 토큰 시퀀스를 UNet Cross-Attention용 [77, 768] 부동소수점 컨디셔닝 텐서로 변환하는 트랜스포머 인코더입니다.
 * WHY: 가짜 감쇠 수식을 박멸하고, 실제 텍스트 프롬프트로부터 의미론적 잠재 컨텍스트 벡터를 생성하기 위해 존재합니다.
 * HOW: Token+Position Embedding -> 12계층 Transformer (LayerNorm -> MultiHead Causal Self-Attention -> QuickGELU MLP) -> Final LayerNorm.
 */
export interface CLIPLayerWeights {
    norm1Gamma: Float32Array;
    norm1Beta: Float32Array;
    qProjWeight: Float32Array;
    qProjBias: Float32Array;
    kProjWeight: Float32Array;
    kProjBias: Float32Array;
    vProjWeight: Float32Array;
    vProjBias: Float32Array;
    outProjWeight: Float32Array;
    outProjBias: Float32Array;
    norm2Gamma: Float32Array;
    norm2Beta: Float32Array;
    mlpFc1Weight: Float32Array;
    mlpFc1Bias: Float32Array;
    mlpFc2Weight: Float32Array;
    mlpFc2Bias: Float32Array;
}
export interface CLIPTextEncoderWeights {
    tokenEmbedding: Float32Array;
    positionEmbedding: Float32Array;
    layers: CLIPLayerWeights[];
    finalNormGamma: Float32Array;
    finalNormBeta: Float32Array;
}
export declare class CLIPTextEncoder {
    static readonly EMBED_DIM = 768;
    static readonly SEQ_LEN = 77;
    static readonly NUM_HEADS = 12;
    static readonly HEAD_DIM = 64;
    /**
     * LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
     */
    static layerNorm(x: Float32Array, seqLen: number, dim: number, gamma: Float32Array, beta: Float32Array, eps?: number): Float32Array;
    /**
     * QuickGELU: x * sigmoid(1.702 * x)
     */
    static quickGELU(x: Float32Array): Float32Array;
    /**
     * Multi-Head Causal Self-Attention (12 heads, 768 dim, Causal Mask)
     */
    static forwardCausalAttention(x: Float32Array, seqLen: number, dim: number, numHeads: number, qW: Float32Array, qB: Float32Array, kW: Float32Array, kB: Float32Array, vW: Float32Array, vB: Float32Array, outW: Float32Array, outB: Float32Array): Float32Array;
    /**
     * Dense Linear: y = x W^T + b
     */
    static linear(x: Float32Array, seqLen: number, inDim: number, outDim: number, w: Float32Array, b?: Float32Array): Float32Array;
    /**
     * CLIP 텍스트 인코더 전체 순전파:
     * [77] 토큰 ID -> Token/Position Embedding -> 12 Transformer Layers -> Final LayerNorm -> [77, 768] 부동소수점 텐서
     */
    static forward(tokenIds: Int32Array, weights: CLIPTextEncoderWeights): Float32Array;
}
