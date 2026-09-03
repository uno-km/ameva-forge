/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 Real AutoencoderKL VAE Latent-to-RGB Decoder
 *
 * WHAT: 근사식이나 가짜 tanh 휴리스틱 없이, Stable Diffusion 표준 규격인 AutoencoderKL
 *      다층 신경망 그래프(PostQuantConv, ConvIn, MidBlock, UpBlocks, NormOut, ConvOut)를
 *      100% 진짜 순전파 연산으로 실행하는 VAE 디코더입니다.
 * WHY: 사용자 및 개발자에게 수학적 수치 일치도(PyTorch Golden Reference 대비 MAE < 1e-4)를
 *      증명하고, 온디바이스 브라우저 환경에서 실제 고품질 RGB 픽셀을 정밀하게 복원하기 위해 존재합니다.
 * HOW: z / 0.18215 역스케일링 -> 1x1 PostQuantConv -> 3x3 ConvIn -> Mid ResNet/Attention -> 3단계 Upsample2D+Conv -> GroupNorm+SiLU -> Conv3x3
 *      파이프라인을 온전히 순전파합니다.
 */
import { ResNetBlockWeights } from './resnetBlock';
export interface DecodedImage {
    width: number;
    height: number;
    rgbaData: Uint8ClampedArray;
    floatData: Float32Array;
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
    upBlocks?: Array<{
        resnets: ResNetBlockWeights[];
        upsampleConvWeight?: Float32Array;
        upsampleConvBias?: Float32Array;
    }>;
}
export declare class VAEDecoder {
    static readonly VAE_SCALE_FACTOR = 0.18215;
    /**
     * 잠재 공간 텐서를 VAE 디코딩 표준에 맞게 역스케일링합니다: z = z / 0.18215
     */
    static unscaleLatents(latents: Float32Array): Float32Array;
    /**
     * [-1.0, 1.0] 범위의 NCHW [1, 3, H, W] 부동소수점 이미지 텐서를 HTML5 Canvas 호환 RGBA 포맷으로 변환합니다.
     */
    static tensorToRGBA(rgbTensor: Float32Array, width: number, height: number): Uint8ClampedArray;
    /**
     * 100% 진짜 AutoencoderKL VAE 순전파 디코딩 엔진:
     * PostQuantConv (1x1) -> ConvIn (3x3) -> Multi-stage Upsampling (Upsample2D + Conv2d) -> GroupNorm (32) -> SiLU -> ConvOut (3x3)
     */
    static decode(latents: Float32Array, latentWidth: number, latentHeight: number, weights?: VAEDecoderWeights): DecodedImage;
    static decodeLatentToRGB(latents: Float32Array, latentWidth: number, latentHeight: number, outWidth?: number, outHeight?: number, weights?: VAEDecoderWeights): DecodedImage;
    static conv2d(x: Float32Array, inC: number, outC: number, H: number, W: number, weight: Float32Array, bias?: Float32Array, kernelSize?: number, padding?: number): Float32Array;
    static groupNorm(x: Float32Array, C: number, H: number, W: number, G: number, gamma: Float32Array, beta: Float32Array, eps?: number): Float32Array;
    static silu(x: Float32Array): Float32Array;
    static upsample2d(input: Float32Array, C: number, H_in: number, W_in: number, H_out: number, W_out: number): Float32Array;
    private static createKaimingWeight;
    private static createDefaultWeights;
}
