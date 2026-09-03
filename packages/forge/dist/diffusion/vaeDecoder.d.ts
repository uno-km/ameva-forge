/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Latent-to-RGB Image Decoder & Canvas Exporter
 *
 * WHAT: 디퓨전 잠재 공간(Latent: [1, 4, 64, 64])을 RGB 픽셀 맵([1, 3, 512, 512])으로 복원하고
 *      HTML5 <canvas> ImageData 포맷(RGBA)으로 즉시 렌더링하는 디코더 파이프라인입니다.
 * WHY: 복원된 고해상도 텐서를 브라우저 화면에 실시간으로 표시하고 PNG 파일로 즉시 추출하기 위해 존재합니다.
 * HOW: VAE 역스케일링 팩터(1.0 / 0.18215)를 적용하고, 8배 공간 업샘플링(3단계 Upsample2D)과
 *      [-1.0, 1.0] -> [0, 255] RGB 정규화 및 RGBA 픽셀 배열 변환을 수행합니다.
 */
export interface DecodedImage {
    width: number;
    height: number;
    rgbaData: Uint8ClampedArray;
    floatData: Float32Array;
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
     * 4채널 잠재 텐서를 3단계 공간 보간을 거쳐 3채널 RGB 픽셀 맵으로 복원하는 CPU/GPU 하이브리드 프로토타입
     */
    static decodeLatentToRGB(latent: Float32Array, latentWidth: number, latentHeight: number, outWidth?: number, outHeight?: number): DecodedImage;
}
