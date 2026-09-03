/**
 * 파일 생성일: 2026-09-03
 * 수정일: 2026-09-03 (P0 긴급 시정: 침묵 폴백 전면 적출, 엄격한 가중치/형상 검증 및 NaN 은폐 방지 도입)
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Latent-to-RGB Decoder Prototype
 *
 * WHAT: VAE 잠재 공간 텐서를 RGB 픽셀로 변환하는 간이 3단계 업샘플링 디코더 프로토타입입니다.
 *      (주의: AutoencoderKL의 MidBlock, Spatial Attention, ResNet UpBlock 계층 및 동적 채널 확장은 아직 미구현 상태입니다.)
 * WHY: 침묵 폴백(Silent Fallback)이나 가짜 가중치 자동 생성을 원천 차단하고,
 *      가중치 누락이나 결함 발생 시 즉각 실패(Fail-Fast)하도록 엄격한 검증을 적용하기 위해 존재합니다.
 * HOW: PostQuantConv (1x1) -> ConvIn (3x3) -> 3단계 Upsample2D+Conv2d -> GroupNorm+SiLU -> ConvOut (3x3) 순으로 실행하며,
 *      모든 텐서 크기 및 Finite 조건을 엄격히 검증합니다.
 */
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
    static readonly DEFAULT_SCALE_FACTOR = 0.18215;
    /**
     * 잠재 공간 텐서를 역스케일링합니다: z / scalingFactor
     */
    static unscaleLatents(latents: Float32Array, scaleFactor?: number): Float32Array;
    /**
     * [-1.0, 1.0] 범위의 NCHW [1, 3, H, W] 부동소수점 이미지 텐서를 HTML5 Canvas 호환 RGBA 포맷으로 변환합니다.
     */
    static tensorToRGBA(rgbTensor: Float32Array, width: number, height: number): Uint8ClampedArray;
    /**
     * 3단계 업샘플링 디코더 순전파:
     * 가중치가 누락되었을 때 어떠한 가짜 가중치도 자동 생성하지 않고 즉각 예외를 분출합니다.
     */
    static decode(latents: Float32Array, latentWidth: number, latentHeight: number, weights: VAEDecoderWeights, scaleFactor?: number): DecodedImage;
    /**
     * decode()의 별칭이며, 요청된 outWidth, outHeight가 실제 출력 크기와 불일치할 경우 즉각 예외를 발생시킵니다.
     */
    static decodeLatentToRGB(latents: Float32Array, latentWidth: number, latentHeight: number, outWidth: number, outHeight: number, weights: VAEDecoderWeights, scaleFactor?: number): DecodedImage;
    static conv2d(x: Float32Array, inC: number, outC: number, H: number, W: number, weight: Float32Array, bias?: Float32Array, kernelSize?: number, padding?: number): Float32Array;
    static groupNorm(x: Float32Array, C: number, H: number, W: number, G: number, gamma: Float32Array, beta: Float32Array, eps?: number): Float32Array;
    static silu(x: Float32Array): Float32Array;
    static upsample2d(input: Float32Array, C: number, H_in: number, W_in: number, H_out: number, W_out: number): Float32Array;
}
/**
 * 테스트 스위트 전용 가중치 생성 유틸리티 (운영 코드에서는 호출 불가하도록 분리 격리)
 */
export declare class VAEDecoderTestFixtures {
    static createSyntheticWeights(inC?: number, midC?: number): VAEDecoderWeights;
}
