/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 Full AutoencoderKL VAE Decoder Architecture
 *
 * WHAT: Stable Diffusion 표준 규격인 AutoencoderKL 다층 신경망 그래프
 *      (PostQuantConv -> ConvIn -> MidBlock(ResNet + Attention + ResNet) -> 4-Stage UpBlocks -> NormOut -> ConvOut)
 *      전수 계층을 100% 진짜 순전파 연산으로 실행하는 고정밀 VAE 디코더입니다.
 * WHY: 침묵 가짜 가중치나 간이 3단계를 넘어, 실제 SD 체크포인트의 계층별 채널 전이(512 -> 512 -> 256 -> 128)와
 *      Spatial Self-Attention을 온전히 지원하는 실체 있는 아키텍처를 제공하기 위해 존재합니다.
 * HOW: 모든 가중치와 입력 형상을 사전에 엄격 검증(Fail-Fast)하고,
 *      Two-pass GroupNorm, Clamped SiLU, Same-Padding Conv2d, Dot-Product Attention을 차례로 순전파합니다.
 */
import { DecodedImage } from './vaeDecoder';
import { ResNetBlockWeights } from './resnetBlock';
export interface AutoencoderKLCapability {
    readonly component: string;
    readonly architecture: string;
    readonly autoencoder_kl_compatible: boolean;
    readonly spatial_self_attention_supported: boolean;
    readonly multi_stage_channel_transition_supported: boolean;
    readonly numerical_parity_verified: boolean;
}
export declare const AUTOENCODER_KL_CAPABILITY: AutoencoderKLCapability;
export interface SpatialSelfAttentionWeights {
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
export interface AutoencoderKLMidBlockWeights {
    resnet1: ResNetBlockWeights;
    attention: SpatialSelfAttentionWeights;
    resnet2: ResNetBlockWeights;
}
export interface AutoencoderKLUpBlockWeights {
    resnets: ResNetBlockWeights[];
    hasUpsample: boolean;
    upsampleConvWeight?: Float32Array;
    upsampleConvBias?: Float32Array;
}
export interface AutoencoderKLWeights {
    postQuantConvWeight: Float32Array;
    postQuantConvBias?: Float32Array;
    convInWeight: Float32Array;
    convInBias?: Float32Array;
    midBlock: AutoencoderKLMidBlockWeights;
    upBlocks: AutoencoderKLUpBlockWeights[];
    normOutGamma: Float32Array;
    normOutBeta: Float32Array;
    convOutWeight: Float32Array;
    convOutBias?: Float32Array;
}
export declare class AutoencoderKLDecoder {
    static readonly DEFAULT_SCALE_FACTOR = 0.18215;
    /**
     * Spatial Self-Attention 순전파:
     * GroupNorm(32) -> Q, K, V 1x1 Conv -> Softmax(Q K^T / sqrt(C)) -> Context -> Out 1x1 Conv -> Residual Skip
     */
    static forwardAttention(x: Float32Array, C: number, H: number, W: number, weights: SpatialSelfAttentionWeights): Float32Array;
    /**
     * 100% 완전한 AutoencoderKL VAE 디코더 순전파:
     * PostQuantConv -> ConvIn -> MidBlock -> 4단계 UpBlocks -> NormOut -> ConvOut
     */
    static decode(latents: Float32Array, latentWidth: number, latentHeight: number, weights: AutoencoderKLWeights, scaleFactor?: number): DecodedImage;
    static conv2d(x: Float32Array, inC: number, outC: number, H: number, W: number, weight: Float32Array, bias?: Float32Array, kernelSize?: number, padding?: number): Float32Array;
    static groupNorm(x: Float32Array, C: number, H: number, W: number, G: number, gamma: Float32Array, beta: Float32Array, eps?: number): Float32Array;
    static silu(x: Float32Array): Float32Array;
    static upsample2d(input: Float32Array, C: number, H_in: number, W_in: number, H_out: number, W_out: number): Float32Array;
}
