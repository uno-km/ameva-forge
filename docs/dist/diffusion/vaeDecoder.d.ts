/**
 * 파일 생성일: 2026-09-03
 * 수정일: 2026-09-03 (P0/P1 엄격 규격 준수: 오류 코드 도입, 3-stage 엄밀 계약, Conv2d 계약 강제, 안정적 2-Pass 분산, Fixture 분리)
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Latent-to-RGB Decoder Prototype
 *
 * WHAT: VAE 잠재 공간 텐서를 RGB 픽셀로 변환하는 간이 3단계 업샘플링 디코더 프로토타입입니다.
 *      (주의: AutoencoderKL의 MidBlock, Spatial Attention, ResNet UpBlock 계층 및 동적 채널 확장은 미구현 상태입니다.)
 * WHY: 침묵 폴백(Silent Fallback)이나 가짜 가중치 자동 생성을 원천 차단하고,
 *      오류 코드(VAEDecoderErrorCode) 기반의 엄격한 계약(Fail-Fast)을 적용하기 위해 존재합니다.
 * HOW: PostQuantConv (1x1) -> ConvIn (3x3) -> 3단계 Upsample2D+Conv2d -> GroupNorm(Two-pass)+SiLU -> ConvOut (3x3) 순으로 실행합니다.
 */
export declare enum VAEDecoderErrorCode {
    VAE_WEIGHTS_REQUIRED = "VAE_WEIGHTS_REQUIRED",
    VAE_WEIGHT_SHAPE_MISMATCH = "VAE_WEIGHT_SHAPE_MISMATCH",
    VAE_NON_FINITE_INPUT = "VAE_NON_FINITE_INPUT",
    VAE_NON_FINITE_WEIGHT = "VAE_NON_FINITE_WEIGHT",
    VAE_NON_FINITE_OUTPUT = "VAE_NON_FINITE_OUTPUT",
    VAE_UPBLOCK_COUNT_MISMATCH = "VAE_UPBLOCK_COUNT_MISMATCH",
    VAE_GROUP_COUNT_INVALID = "VAE_GROUP_COUNT_INVALID",
    VAE_GROUP_DIVISIBILITY_ERROR = "VAE_GROUP_DIVISIBILITY_ERROR",
    VAE_SCALE_FACTOR_INVALID = "VAE_SCALE_FACTOR_INVALID",
    VAE_OUTPUT_SCALE_MISMATCH = "VAE_OUTPUT_SCALE_MISMATCH",
    VAE_RESOURCE_LIMIT_EXCEEDED = "VAE_RESOURCE_LIMIT_EXCEEDED",
    VAE_CONV_CONTRACT_INVALID = "VAE_CONV_CONTRACT_INVALID",
    VAE_INVALID_DIMENSION = "VAE_INVALID_DIMENSION",
    VAE_EPS_INVALID = "VAE_EPS_INVALID"
}
export declare class VAEDecoderError extends Error {
    readonly code: VAEDecoderErrorCode;
    constructor(code: VAEDecoderErrorCode, message: string);
}
export interface VAEDecoderArchitecture {
    readonly latentChannels: number;
    readonly midChannels: number;
    readonly upBlockCount: number;
    readonly normGroups: number;
    readonly defaultScaleFactor: number;
    readonly upsampleFactor: number;
    readonly convKernelSize: number;
}
export declare const VAE_DECODER_ARCHITECTURE: VAEDecoderArchitecture;
export interface VAEDecoderCapability {
    readonly component: string;
    readonly architecture: string;
    readonly autoencoder_kl_compatible: boolean;
    readonly supports_real_checkpoint: boolean;
    readonly numerical_parity_verified: boolean;
}
export declare const VAE_DECODER_CAPABILITY: VAEDecoderCapability;
export interface VAEDecoderLimits {
    readonly maxTensorElements?: number;
    readonly maxOutputPixels?: number;
    readonly maxWeightElements?: number;
}
export interface DecodedImage {
    width: number;
    height: number;
    rgbaData: Uint8ClampedArray;
    floatData: Float32Array;
}
export interface VAEStageWeights {
    upsampleConvWeight: Float32Array;
    upsampleConvBias?: Float32Array;
    normGamma: Float32Array;
    normBeta: Float32Array;
}
export interface VAEDecoderWeights {
    postQuantConvWeight: Float32Array;
    postQuantConvBias?: Float32Array;
    convInWeight: Float32Array;
    convInBias?: Float32Array;
    normOutGamma: Float32Array;
    normOutBeta: Float32Array;
    convOutWeight: Float32Array;
    convOutBias?: Float32Array;
    upBlocks: VAEStageWeights[];
}
export declare class VAEDecoder {
    static readonly DEFAULT_SCALE_FACTOR: number;
    /**
     * 잠재 공간 텐서를 역스케일링합니다: z / scalingFactor
     */
    static unscaleLatents(latents: Float32Array, scaleFactor?: number): Float32Array;
    /**
     * [-1.0, 1.0] 범위의 NCHW [1, 3, H, W] 부동소수점 이미지 텐서를 HTML5 Canvas 호환 RGBA 포맷으로 변환합니다.
     */
    static tensorToRGBA(rgbTensor: Float32Array, width: number, height: number, limits?: VAEDecoderLimits): Uint8ClampedArray;
    /**
     * 3단계 업샘플링 디코더 순전파:
     * 가중치 필수 검증, 사전 유한성 검증, 정확한 3-stage 검증 및 리소스 한계를 집행합니다.
     */
    static decode(latents: Float32Array, latentWidth: number, latentHeight: number, weights: VAEDecoderWeights, scaleFactor?: number, limits?: VAEDecoderLimits): DecodedImage;
    /**
     * decode()의 별칭이며, 요청된 outWidth, outHeight가 실제 출력 크기와 불일치할 경우 즉각 예외를 발생시킵니다.
     */
    static decodeLatentToRGB(latents: Float32Array, latentWidth: number, latentHeight: number, outWidth: number, outHeight: number, weights: VAEDecoderWeights, scaleFactor?: number, limits?: VAEDecoderLimits): DecodedImage;
    static conv2d(x: Float32Array, inC: number, outC: number, H: number, W: number, weight: Float32Array, bias?: Float32Array, kernelSize?: number, padding?: number): Float32Array;
    static groupNorm(x: Float32Array, C: number, H: number, W: number, G: number, gamma: Float32Array, beta: Float32Array, eps?: number): Float32Array;
    static silu(x: Float32Array): Float32Array;
    static upsample2d(input: Float32Array, C: number, H_in: number, W_in: number, H_out: number, W_out: number): Float32Array;
}
