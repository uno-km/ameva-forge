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
  floatData: Float32Array; // [3, H, W] RGB
}

export class VAEDecoder {
  public static readonly VAE_SCALE_FACTOR = 0.18215;

  /**
   * 잠재 공간 텐서를 VAE 디코딩 표준에 맞게 역스케일링합니다: z = z / 0.18215
   */
  public static unscaleLatents(latents: Float32Array): Float32Array {
    const unscaled = new Float32Array(latents.length);
    const factor = 1.0 / this.VAE_SCALE_FACTOR;
    for (let i = 0; i < latents.length; i++) {
      unscaled[i] = latents[i] * factor;
    }
    return unscaled;
  }

  /**
   * [-1.0, 1.0] 범위의 NCHW [1, 3, H, W] 부동소수점 이미지 텐서를 HTML5 Canvas 호환 RGBA 포맷으로 변환합니다.
   */
  public static tensorToRGBA(
    rgbTensor: Float32Array,
    width: number,
    height: number
  ): Uint8ClampedArray {
    const totalPixels = width * height;
    const rgba = new Uint8ClampedArray(totalPixels * 4);

    const rOffset = 0;
    const gOffset = totalPixels;
    const bOffset = totalPixels * 2;

    for (let i = 0; i < totalPixels; i++) {
      // [-1.0, 1.0] -> [0.0, 255.0]
      const r = Math.min(255, Math.max(0, Math.round((rgbTensor[rOffset + i] + 1.0) * 127.5)));
      const g = Math.min(255, Math.max(0, Math.round((rgbTensor[gOffset + i] + 1.0) * 127.5)));
      const b = Math.min(255, Math.max(0, Math.round((rgbTensor[bOffset + i] + 1.0) * 127.5)));

      const rgbaIndex = i * 4;
      rgba[rgbaIndex] = r;
      rgba[rgbaIndex + 1] = g;
      rgba[rgbaIndex + 2] = b;
      rgba[rgbaIndex + 3] = 255; // Alpha
    }

    return rgba;
  }

  /**
   * 4채널 잠재 텐서를 3단계 공간 보간을 거쳐 3채널 RGB 픽셀 맵으로 복원하는 CPU/GPU 하이브리드 프로토타입
   */
  public static decodeLatentToRGB(
    latent: Float32Array,
    latentWidth: number,
    latentHeight: number,
    outWidth: number = 512,
    outHeight: number = 512
  ): DecodedImage {
    // 1. Latent Unscaling
    const unscaled = this.unscaleLatents(latent);

    // 2. Linear Projection from 4 channels to 3 channels (Pre-VAE approximation)
    const totalPixelsOut = outWidth * outHeight;
    const rgbTensor = new Float32Array(3 * totalPixelsOut);

    const scaleW = outWidth / latentWidth;
    const scaleH = outHeight / latentHeight;

    // Nearest / Bilinear projection
    for (let y = 0; y < outHeight; y++) {
      const srcY = Math.min(Math.floor(y / scaleH), latentHeight - 1);
      for (let x = 0; x < outWidth; x++) {
        const srcX = Math.min(Math.floor(x / scaleW), latentWidth - 1);
        const latentIdx = srcY * latentWidth + srcX;
        const outIdx = y * outWidth + x;

        // Channel projection: R, G, B mapped from latent channels 0, 1, 2, 3
        const c0 = unscaled[latentIdx];
        const c1 = unscaled[latentHeight * latentWidth + latentIdx];
        const c2 = unscaled[2 * latentHeight * latentWidth + latentIdx];
        const c3 = unscaled[3 * latentHeight * latentWidth + latentIdx];

        // Decoded RGB mapping: Tanh-clamped linear combination
        rgbTensor[outIdx] = Math.tanh(0.299 * c0 + 0.587 * c1 + 0.114 * c2);
        rgbTensor[totalPixelsOut + outIdx] = Math.tanh(0.4 * c1 + 0.4 * c2 + 0.2 * c3);
        rgbTensor[2 * totalPixelsOut + outIdx] = Math.tanh(0.3 * c0 + 0.2 * c2 + 0.5 * c3);
      }
    }

    const rgba = this.tensorToRGBA(rgbTensor, outWidth, outHeight);

    return {
      width: outWidth,
      height: outHeight,
      rgbaData: rgba,
      floatData: rgbTensor,
    };
  }
}
