/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-332 UNet Denoising Neural Network Execution Graph
 *
 * WHAT: 시간 임베딩, 다운블록, 미드블록, 업블록 및 텍스트 교차 어텐션(Cross-Attention)을
 *      하나의 유기적인 순전파 신경망 그래프로 실행하는 UNet 엔진입니다.
 * WHY: 가짜 감쇠 수식을 영구 박멸하고, 실제 잠재 노이즈와 텍스트 임베딩으로부터 노이즈 잔차(Residual)를 예측하기 위해 존재합니다.
 * HOW: Sinusoidal TimeEmbedding -> DownBlocks(ResNet + CrossAttn) -> MidBlock -> UpBlocks(Upsample + Skip Concat + ResNet + CrossAttn) -> OutConv.
 */
import { ResNetBlockWeights } from './resnetBlock';
export interface SpatialCrossAttentionWeights {
    normGamma: Float32Array;
    normBeta: Float32Array;
    qWeight: Float32Array;
    qBias?: Float32Array;
    kWeight: Float32Array;
    kBias?: Float32Array;
    vWeight: Float32Array;
    vBias?: Float32Array;
    outWeight: Float32Array;
    outBias?: Float32Array;
}
export interface UNetBlockWeights {
    resnets: ResNetBlockWeights[];
    attentions: SpatialCrossAttentionWeights[];
    downsampleConvWeight?: Float32Array;
    downsampleConvBias?: Float32Array;
    upsampleConvWeight?: Float32Array;
    upsampleConvBias?: Float32Array;
}
export interface UNetWeights {
    convInWeight: Float32Array;
    convInBias?: Float32Array;
    timeMlp1Weight: Float32Array;
    timeMlp1Bias: Float32Array;
    timeMlp2Weight: Float32Array;
    timeMlp2Bias: Float32Array;
    downBlocks: UNetBlockWeights[];
    midBlock: {
        resnet1: ResNetBlockWeights;
        attention: SpatialCrossAttentionWeights;
        resnet2: ResNetBlockWeights;
    };
    upBlocks: UNetBlockWeights[];
    normOutGamma: Float32Array;
    normOutBeta: Float32Array;
    convOutWeight: Float32Array;
    convOutBias?: Float32Array;
}
export declare class UNetGraph {
    static readonly TIME_DIM = 320;
    /**
     * 정현파(Sinusoidal) 시간 임베딩 계산:
     * PE(t, 2i) = sin(t / 10000^(2i/d)), PE(t, 2i+1) = cos(t / 10000^(2i/d))
     */
    static computeSinusoidalTimeEmbedding(timestep: number, dim?: number): Float32Array;
    /**
     * Spatial Cross-Attention:
     * Latent Q와 텍스트 임베딩 K, V 사이의 행렬 곱셈을 통한 의미론적 조건 주입
     */
    static forwardCrossAttention(x: Float32Array, C: number, H: number, W: number, context: Float32Array, // [77, textDim]
    textSeqLen: number, textDim: number, weights: SpatialCrossAttentionWeights): Float32Array;
    /**
     * UNet 디노이징 신경망 전체 순전파:
     * 잠재 텐서(z_t) + 타임스텝(t) + 텍스트 컨텍스트 임베딩(c) -> 예측 노이즈(eps_theta)
     */
    static forward(sample: Float32Array, timestep: number, textContext: Float32Array, // [77, 768]
    weights: UNetWeights, height?: number, width?: number, baseChannels?: number): Float32Array;
}
