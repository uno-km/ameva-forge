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

export enum VAEDecoderErrorCode {
  VAE_WEIGHTS_REQUIRED = 'VAE_WEIGHTS_REQUIRED',
  VAE_WEIGHT_SHAPE_MISMATCH = 'VAE_WEIGHT_SHAPE_MISMATCH',
  VAE_NON_FINITE_INPUT = 'VAE_NON_FINITE_INPUT',
  VAE_NON_FINITE_WEIGHT = 'VAE_NON_FINITE_WEIGHT',
  VAE_NON_FINITE_OUTPUT = 'VAE_NON_FINITE_OUTPUT',
  VAE_UPBLOCK_COUNT_MISMATCH = 'VAE_UPBLOCK_COUNT_MISMATCH',
  VAE_GROUP_COUNT_INVALID = 'VAE_GROUP_COUNT_INVALID',
  VAE_GROUP_DIVISIBILITY_ERROR = 'VAE_GROUP_DIVISIBILITY_ERROR',
  VAE_SCALE_FACTOR_INVALID = 'VAE_SCALE_FACTOR_INVALID',
  VAE_OUTPUT_SCALE_MISMATCH = 'VAE_OUTPUT_SCALE_MISMATCH',
  VAE_RESOURCE_LIMIT_EXCEEDED = 'VAE_RESOURCE_LIMIT_EXCEEDED',
  VAE_CONV_CONTRACT_INVALID = 'VAE_CONV_CONTRACT_INVALID',
  VAE_INVALID_DIMENSION = 'VAE_INVALID_DIMENSION',
  VAE_EPS_INVALID = 'VAE_EPS_INVALID',
}

export class VAEDecoderError extends Error {
  public readonly code: VAEDecoderErrorCode;

  constructor(code: VAEDecoderErrorCode, message: string) {
    super(`[VAEDecoder:${code}] ${message}`);
    this.name = 'VAEDecoderError';
    this.code = code;
    Object.setPrototypeOf(this, new.target.prototype);
  }
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

export const VAE_DECODER_ARCHITECTURE: VAEDecoderArchitecture = Object.freeze({
  latentChannels: 4,
  midChannels: 32,
  upBlockCount: 3,
  normGroups: 32,
  defaultScaleFactor: 0.18215,
  upsampleFactor: 8,
  convKernelSize: 3,
});

export interface VAEDecoderCapability {
  readonly component: string;
  readonly architecture: string;
  readonly autoencoder_kl_compatible: boolean;
  readonly supports_real_checkpoint: boolean;
  readonly numerical_parity_verified: boolean;
}

export const VAE_DECODER_CAPABILITY: VAEDecoderCapability = Object.freeze({
  component: 'vae-decoder-prototype',
  architecture: 'fixed-3-stage-convolutional',
  autoencoder_kl_compatible: false,
  supports_real_checkpoint: false,
  numerical_parity_verified: false,
});

export interface VAEDecoderLimits {
  readonly maxTensorElements?: number;
  readonly maxOutputPixels?: number;
  readonly maxWeightElements?: number;
}

const DEFAULT_LIMITS = {
  maxTensorElements: 16 * 1024 * 1024,
  maxOutputPixels: 2048 * 2048,
  maxWeightElements: 64 * 1024 * 1024,
};

export interface DecodedImage {
  width: number;
  height: number;
  rgbaData: Uint8ClampedArray;
  floatData: Float32Array; // [3, H, W] RGB
}

export interface VAEStageWeights {
  upsampleConvWeight: Float32Array; // [currentC, currentC, 3, 3]
  upsampleConvBias?: Float32Array;  // [currentC]
  normGamma: Float32Array;          // [currentC]
  normBeta: Float32Array;           // [currentC]
}

export interface VAEDecoderWeights {
  postQuantConvWeight: Float32Array; // [4, 4, 1, 1]
  postQuantConvBias?: Float32Array;  // [4]
  convInWeight: Float32Array;        // [C_mid, 4, 3, 3]
  convInBias?: Float32Array;         // [C_mid]
  normOutGamma: Float32Array;        // [C_mid]
  normOutBeta: Float32Array;         // [C_mid]
  convOutWeight: Float32Array;       // [3, C_mid, 3, 3]
  convOutBias?: Float32Array;        // [3]
  upBlocks: VAEStageWeights[];       // 정확히 3개 업샘플 스테이지 가중치 (필수)
}

function assertPositiveSafeInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new VAEDecoderError(
      VAEDecoderErrorCode.VAE_INVALID_DIMENSION,
      `${name} must be a positive safe integer, received: ${value}`
    );
  }
}

function checkedElementCount(name: string, dimensions: readonly number[], maxLimit: number): number {
  let total = 1;
  for (const d of dimensions) {
    assertPositiveSafeInteger(`${name} dimension`, d);
    if (total > Math.floor(maxLimit / d)) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_RESOURCE_LIMIT_EXCEEDED,
        `${name} element count exceeds maximum limit of ${maxLimit}`
      );
    }
    total *= d;
  }
  return total;
}

function assertLength(name: string, actual: number, expected: number): void {
  if (actual !== expected) {
    throw new VAEDecoderError(
      VAEDecoderErrorCode.VAE_WEIGHT_SHAPE_MISMATCH,
      `${name} length mismatch: expected ${expected}, received ${actual}`
    );
  }
}

function assertAllFinite(name: string, values: Float32Array, isWeight: boolean = false): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      const code = isWeight
        ? VAEDecoderErrorCode.VAE_NON_FINITE_WEIGHT
        : VAEDecoderErrorCode.VAE_NON_FINITE_INPUT;
      throw new VAEDecoderError(
        code,
        `${name} contains non-finite value at index ${i}: ${values[i]}`
      );
    }
  }
}

export class VAEDecoder {
  public static readonly DEFAULT_SCALE_FACTOR = VAE_DECODER_ARCHITECTURE.defaultScaleFactor;

  /**
   * 잠재 공간 텐서를 역스케일링합니다: z / scalingFactor
   */
  public static unscaleLatents(
    latents: Float32Array,
    scaleFactor: number = VAEDecoder.DEFAULT_SCALE_FACTOR
  ): Float32Array {
    if (!Number.isFinite(scaleFactor) || scaleFactor <= 0) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_SCALE_FACTOR_INVALID,
        `scaleFactor must be a finite positive number, received: ${scaleFactor}`
      );
    }
    assertAllFinite('latents in unscaleLatents', latents, false);

    const unscaled = new Float32Array(latents.length);
    const factor = 1.0 / scaleFactor;
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
    height: number,
    limits: VAEDecoderLimits = DEFAULT_LIMITS
  ): Uint8ClampedArray {
    assertPositiveSafeInteger('width', width);
    assertPositiveSafeInteger('height', height);

    const maxPixels = limits.maxOutputPixels ?? DEFAULT_LIMITS.maxOutputPixels;
    const totalPixels = checkedElementCount('output canvas pixels', [width, height], maxPixels);

    assertLength('rgbTensor for RGBA conversion', rgbTensor.length, totalPixels * 3);
    assertAllFinite('rgbTensor before RGBA conversion', rgbTensor, false);

    const rgba = new Uint8ClampedArray(totalPixels * 4);
    const rOffset = 0;
    const gOffset = totalPixels;
    const bOffset = totalPixels * 2;

    for (let i = 0; i < totalPixels; i++) {
      const r = Math.min(255, Math.max(0, Math.round((rgbTensor[rOffset + i] + 1.0) * 127.5)));
      const g = Math.min(255, Math.max(0, Math.round((rgbTensor[gOffset + i] + 1.0) * 127.5)));
      const b = Math.min(255, Math.max(0, Math.round((rgbTensor[bOffset + i] + 1.0) * 127.5)));

      const rgbaIndex = i * 4;
      rgba[rgbaIndex] = r;
      rgba[rgbaIndex + 1] = g;
      rgba[rgbaIndex + 2] = b;
      rgba[rgbaIndex + 3] = 255;
    }

    return rgba;
  }

  /**
   * 3단계 업샘플링 디코더 순전파:
   * 가중치 필수 검증, 사전 유한성 검증, 정확한 3-stage 검증 및 리소스 한계를 집행합니다.
   */
  public static decode(
    latents: Float32Array,
    latentWidth: number,
    latentHeight: number,
    weights: VAEDecoderWeights,
    scaleFactor: number = VAEDecoder.DEFAULT_SCALE_FACTOR,
    limits: VAEDecoderLimits = DEFAULT_LIMITS
  ): DecodedImage {
    // 1. 가중치 객체 필수 검증
    if (!weights) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_WEIGHTS_REQUIRED,
        'VAE decoder weights are required. Refusing to decode with synthetic weights.'
      );
    }

    // 2. 정확한 3-stage 검증 (4개 이상 침묵 허용 차단)
    if (!weights.upBlocks || weights.upBlocks.length !== VAE_DECODER_ARCHITECTURE.upBlockCount) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_UPBLOCK_COUNT_MISMATCH,
        `VAE decoder requires exactly ${VAE_DECODER_ARCHITECTURE.upBlockCount} upsample stages in upBlocks, received: ${weights.upBlocks?.length ?? 0}`
      );
    }

    // 3. 차원 정수성 및 리소스 한계 검증
    assertPositiveSafeInteger('latentWidth', latentWidth);
    assertPositiveSafeInteger('latentHeight', latentHeight);

    const maxElements = limits.maxTensorElements ?? DEFAULT_LIMITS.maxTensorElements;
    const inChannels = VAE_DECODER_ARCHITECTURE.latentChannels;
    const expectedLatentLen = checkedElementCount('latent input', [inChannels, latentHeight, latentWidth], maxElements);
    assertLength('latents', latents.length, expectedLatentLen);

    // 4. 입력 텐서 사전 유한성 검증
    assertAllFinite('latents input', latents, false);

    // 5. 가중치 및 편향 전수 사전 유한성 검증 (P0-7: 연산 전 Fail-Fast)
    const cMid = VAE_DECODER_ARCHITECTURE.midChannels;
    assertLength('postQuantConvWeight', weights.postQuantConvWeight.length, inChannels * inChannels * 1 * 1);
    assertAllFinite('postQuantConvWeight', weights.postQuantConvWeight, true);
    if (weights.postQuantConvBias) {
      assertLength('postQuantConvBias', weights.postQuantConvBias.length, inChannels);
      assertAllFinite('postQuantConvBias', weights.postQuantConvBias, true);
    }

    assertLength('convInWeight', weights.convInWeight.length, cMid * inChannels * 3 * 3);
    assertAllFinite('convInWeight', weights.convInWeight, true);
    if (weights.convInBias) {
      assertLength('convInBias', weights.convInBias.length, cMid);
      assertAllFinite('convInBias', weights.convInBias, true);
    }

    assertLength('normOutGamma', weights.normOutGamma.length, cMid);
    assertAllFinite('normOutGamma', weights.normOutGamma, true);
    assertLength('normOutBeta', weights.normOutBeta.length, cMid);
    assertAllFinite('normOutBeta', weights.normOutBeta, true);

    assertLength('convOutWeight', weights.convOutWeight.length, 3 * cMid * 3 * 3);
    assertAllFinite('convOutWeight', weights.convOutWeight, true);
    if (weights.convOutBias) {
      assertLength('convOutBias', weights.convOutBias.length, 3);
      assertAllFinite('convOutBias', weights.convOutBias, true);
    }

    for (let stage = 0; stage < 3; stage++) {
      const stageBlock = weights.upBlocks[stage];
      if (!stageBlock?.upsampleConvWeight) {
        throw new VAEDecoderError(
          VAEDecoderErrorCode.VAE_WEIGHTS_REQUIRED,
          `Missing VAE upsample convolution weight at stage ${stage}`
        );
      }
      if (!stageBlock?.normGamma || !stageBlock?.normBeta) {
        throw new VAEDecoderError(
          VAEDecoderErrorCode.VAE_WEIGHTS_REQUIRED,
          `Missing VAE upsample norm parameters at stage ${stage}`
        );
      }

      assertLength(`upBlocks[${stage}].upsampleConvWeight`, stageBlock.upsampleConvWeight.length, cMid * cMid * 3 * 3);
      assertAllFinite(`upBlocks[${stage}].upsampleConvWeight`, stageBlock.upsampleConvWeight, true);

      if (stageBlock.upsampleConvBias) {
        assertLength(`upBlocks[${stage}].upsampleConvBias`, stageBlock.upsampleConvBias.length, cMid);
        assertAllFinite(`upBlocks[${stage}].upsampleConvBias`, stageBlock.upsampleConvBias, true);
      }

      assertLength(`upBlocks[${stage}].normGamma`, stageBlock.normGamma.length, cMid);
      assertAllFinite(`upBlocks[${stage}].normGamma`, stageBlock.normGamma, true);
      assertLength(`upBlocks[${stage}].normBeta`, stageBlock.normBeta.length, cMid);
      assertAllFinite(`upBlocks[${stage}].normBeta`, stageBlock.normBeta, true);
    }

    // 6. 순전파 실행
    const unscaled = this.unscaleLatents(latents, scaleFactor);

    // Post-Quant Conv (1x1 Conv, 4 -> 4)
    const postQuant = this.conv2d(unscaled, inChannels, inChannels, latentHeight, latentWidth, weights.postQuantConvWeight, weights.postQuantConvBias, 1, 0);
    assertAllFinite('postQuant output', postQuant, false);

    // Conv In (3x3 Conv, 4 -> 32)
    const featIn = this.conv2d(postQuant, inChannels, cMid, latentHeight, latentWidth, weights.convInWeight, weights.convInBias, 3, 1);
    assertAllFinite('featIn output', featIn, false);

    let currentFeat = featIn;
    let currentH = latentHeight;
    let currentW = latentWidth;
    const currentC = cMid;

    // 3단계 업샘플링 (64 -> 128 -> 256 -> 512)
    for (let stage = 0; stage < 3; stage++) {
      const stageBlock = weights.upBlocks[stage];
      const nextH = currentH * 2;
      const nextW = currentW * 2;

      checkedElementCount(`stage ${stage} upsampled`, [currentC, nextH, nextW], maxElements);

      // Upsample2D (Bilinear 2x)
      const upsampled = this.upsample2d(currentFeat, currentC, currentH, currentW, nextH, nextW);
      assertAllFinite(`upsampled stage ${stage}`, upsampled, false);

      // Conv2d (3x3)
      const convOutStage = this.conv2d(upsampled, currentC, currentC, nextH, nextW, stageBlock.upsampleConvWeight, stageBlock.upsampleConvBias, 3, 1);
      assertAllFinite(`convOutStage stage ${stage}`, convOutStage, false);

      // GroupNorm (Two-pass numerically stable) + SiLU
      const normed = this.groupNorm(convOutStage, currentC, nextH, nextW, Math.min(VAE_DECODER_ARCHITECTURE.normGroups, currentC), stageBlock.normGamma, stageBlock.normBeta);
      assertAllFinite(`normed stage ${stage}`, normed, false);

      currentFeat = this.silu(normed);
      assertAllFinite(`silu stage ${stage}`, currentFeat, false);

      currentH = nextH;
      currentW = nextW;
    }

    // Final Output Norm + SiLU
    const finalNormed = this.groupNorm(currentFeat, currentC, currentH, currentW, Math.min(VAE_DECODER_ARCHITECTURE.normGroups, currentC), weights.normOutGamma, weights.normOutBeta);
    assertAllFinite('finalNormed', finalNormed, false);

    const finalAct = this.silu(finalNormed);
    assertAllFinite('finalAct', finalAct, false);

    // Conv Out (3x3 Conv, 32 -> 3 RGB)
    const rgbTensor = this.conv2d(finalAct, currentC, 3, currentH, currentW, weights.convOutWeight, weights.convOutBias, 3, 1);
    assertAllFinite('rgbTensor', rgbTensor, false);

    // Canvas RGBA 변환
    const rgba = this.tensorToRGBA(rgbTensor, currentW, currentH, limits);

    return {
      width: currentW,
      height: currentH,
      rgbaData: rgba,
      floatData: rgbTensor,
    };
  }

  /**
   * decode()의 별칭이며, 요청된 outWidth, outHeight가 실제 출력 크기와 불일치할 경우 즉각 예외를 발생시킵니다.
   */
  public static decodeLatentToRGB(
    latents: Float32Array,
    latentWidth: number,
    latentHeight: number,
    outWidth: number,
    outHeight: number,
    weights: VAEDecoderWeights,
    scaleFactor?: number,
    limits?: VAEDecoderLimits
  ): DecodedImage {
    const expectedW = latentWidth * VAE_DECODER_ARCHITECTURE.upsampleFactor;
    const expectedH = latentHeight * VAE_DECODER_ARCHITECTURE.upsampleFactor;
    if (outWidth !== expectedW || outHeight !== expectedH) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_OUTPUT_SCALE_MISMATCH,
        `decodeLatentToRGB scale mismatch: requested ${outWidth}x${outHeight}, but latent ${latentWidth}x${latentHeight} scales to ${expectedW}x${expectedH}`
      );
    }
    return this.decode(latents, latentWidth, latentHeight, weights, scaleFactor, limits);
  }

  // --- 수치 신경망 기본 연산자 (Same-Padding 엄격 계약 및 2-Pass 분산 탑재) ---

  public static conv2d(
    x: Float32Array,
    inC: number,
    outC: number,
    H: number,
    W: number,
    weight: Float32Array,
    bias?: Float32Array,
    kernelSize: number = 3,
    padding: number = 1
  ): Float32Array {
    assertPositiveSafeInteger('inC', inC);
    assertPositiveSafeInteger('outC', outC);
    assertPositiveSafeInteger('H', H);
    assertPositiveSafeInteger('W', W);

    // P0-4: Same-convolution 전용 엄격 계약 집행
    if (!Number.isInteger(kernelSize) || kernelSize <= 0 || kernelSize % 2 === 0) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_CONV_CONTRACT_INVALID,
        `conv2d requires a positive odd kernel size, received: ${kernelSize}`
      );
    }
    const expectedPadding = Math.floor(kernelSize / 2);
    if (padding !== expectedPadding) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_CONV_CONTRACT_INVALID,
        `conv2d only supports same-padding convolution: kernelSize=${kernelSize}, required padding=${expectedPadding}, received padding=${padding}`
      );
    }

    assertLength('x in conv2d', x.length, inC * H * W);
    assertLength('weight in conv2d', weight.length, outC * inC * kernelSize * kernelSize);
    if (bias) {
      assertLength('bias in conv2d', bias.length, outC);
    }

    const outH = H;
    const outW = W;
    const hw = outH * outW;
    const out = new Float32Array(outC * hw);
    const pad = padding;

    for (let oc = 0; oc < outC; oc++) {
      const b = bias ? bias[oc] : 0.0;
      const ocOffset = oc * hw;

      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let sum = b;

          for (let ic = 0; ic < inC; ic++) {
            const icOffset = ic * (H * W);
            const wOffset = (oc * inC + ic) * (kernelSize * kernelSize);

            for (let kh = 0; kh < kernelSize; kh++) {
              const ih = oh - pad + kh;
              if (ih < 0 || ih >= H) continue;

              for (let kw = 0; kw < kernelSize; kw++) {
                const iw = ow - pad + kw;
                if (iw < 0 || iw >= W) continue;

                const val = x[icOffset + ih * W + iw];
                const w = weight[wOffset + kh * kernelSize + kw];
                sum += val * w;
              }
            }
          }

          out[ocOffset + oh * outW + ow] = sum;
        }
      }
    }

    return out;
  }

  public static groupNorm(
    x: Float32Array,
    C: number,
    H: number,
    W: number,
    G: number,
    gamma: Float32Array,
    beta: Float32Array,
    eps: number = 1e-5
  ): Float32Array {
    assertPositiveSafeInteger('C', C);
    assertPositiveSafeInteger('H', H);
    assertPositiveSafeInteger('W', W);

    if (!Number.isInteger(G) || G <= 0) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_GROUP_COUNT_INVALID,
        `GroupNorm group count must be positive safe integer: ${G}`
      );
    }
    if (C % G !== 0) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_GROUP_DIVISIBILITY_ERROR,
        `GroupNorm requires C divisible by G: C=${C}, G=${G}`
      );
    }
    if (!Number.isFinite(eps) || eps <= 0) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_EPS_INVALID,
        `GroupNorm eps must be a finite positive number, received: ${eps}`
      );
    }
    if (gamma.length !== C || beta.length !== C) {
      throw new VAEDecoderError(
        VAEDecoderErrorCode.VAE_WEIGHT_SHAPE_MISMATCH,
        `GroupNorm affine parameter mismatch: C=${C}, gamma=${gamma.length}, beta=${beta.length}`
      );
    }
    assertLength('x in groupNorm', x.length, C * H * W);

    const hw = H * W;
    const channelsPerGroup = C / G;
    const groupSize = channelsPerGroup * hw;
    const out = new Float32Array(x.length);

    for (let g = 0; g < G; g++) {
      const baseC = g * channelsPerGroup;

      // P1-1: Two-pass variance algorithm (수치 안정화: 큰 Offset 및 작은 분산의 상쇄 오차 방지)
      let sum = 0.0;
      for (let c = 0; c < channelsPerGroup; c++) {
        const cIdx = (baseC + c) * hw;
        for (let i = 0; i < hw; i++) {
          sum += x[cIdx + i];
        }
      }
      const mean = sum / groupSize;

      let sqDiffSum = 0.0;
      for (let c = 0; c < channelsPerGroup; c++) {
        const cIdx = (baseC + c) * hw;
        for (let i = 0; i < hw; i++) {
          const diff = x[cIdx + i] - mean;
          sqDiffSum += diff * diff;
        }
      }
      const variance = sqDiffSum / groupSize;
      const invStd = 1.0 / Math.sqrt(variance + eps);

      for (let c = 0; c < channelsPerGroup; c++) {
        const actualC = baseC + c;
        const cIdx = actualC * hw;
        const scale = gamma[actualC];
        const shift = beta[actualC];

        for (let i = 0; i < hw; i++) {
          const normX = (x[cIdx + i] - mean) * invStd;
          out[cIdx + i] = normX * scale + shift;
        }
      }
    }

    return out;
  }

  public static silu(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      const clamped = Math.max(-88.0, Math.min(88.0, v));
      const sig = 1.0 / (1.0 + Math.exp(-clamped));
      out[i] = v * sig;
    }
    return out;
  }

  public static upsample2d(
    input: Float32Array,
    C: number,
    H_in: number,
    W_in: number,
    H_out: number,
    W_out: number
  ): Float32Array {
    assertPositiveSafeInteger('C', C);
    assertPositiveSafeInteger('H_in', H_in);
    assertPositiveSafeInteger('W_in', W_in);
    assertPositiveSafeInteger('H_out', H_out);
    assertPositiveSafeInteger('W_out', W_out);

    assertLength('input in upsample2d', input.length, C * H_in * W_in);

    const out = new Float32Array(C * H_out * W_out);
    const scale_h = H_out / H_in;
    const scale_w = W_out / W_in;

    for (let c = 0; c < C; c++) {
      const in_c_offset = c * (H_in * W_in);
      const out_c_offset = c * (H_out * W_out);

      for (let h_out = 0; h_out < H_out; h_out++) {
        const real_h = (h_out + 0.5) / scale_h - 0.5;
        const h0 = Math.max(0, Math.min(Math.floor(real_h), H_in - 1));
        const h1 = Math.min(h0 + 1, H_in - 1);
        const dh = Math.max(0.0, Math.min(1.0, real_h - h0));

        for (let w_out = 0; w_out < W_out; w_out++) {
          const real_w = (w_out + 0.5) / scale_w - 0.5;
          const w0 = Math.max(0, Math.min(Math.floor(real_w), W_in - 1));
          const w1 = Math.min(w0 + 1, W_in - 1);
          const dw = Math.max(0.0, Math.min(1.0, real_w - w0));

          const v00 = input[in_c_offset + h0 * W_in + w0];
          const v01 = input[in_c_offset + h0 * W_in + w1];
          const v10 = input[in_c_offset + h1 * W_in + w0];
          const v11 = input[in_c_offset + h1 * W_in + w1];

          const top = v00 * (1.0 - dw) + v01 * dw;
          const bottom = v10 * (1.0 - dw) + v11 * dw;

          out[out_c_offset + h_out * W_out + w_out] = top * (1.0 - dh) + bottom * dh;
        }
      }
    }

    return out;
  }
}
