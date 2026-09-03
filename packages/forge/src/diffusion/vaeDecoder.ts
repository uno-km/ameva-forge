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
  upBlocks: VAEStageWeights[];       // 3개 업샘플 스테이지 가중치 (필수)
}

function assertLength(name: string, actual: number, expected: number): void {
  if (actual !== expected) {
    throw new Error(`[VAEDecoder] ${name} length mismatch: expected ${expected}, received ${actual}`);
  }
}

function assertAllFinite(name: string, values: Float32Array): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`[VAEDecoder] ${name} contains non-finite value at index ${i}: ${values[i]}`);
    }
  }
}

export class VAEDecoder {
  public static readonly DEFAULT_SCALE_FACTOR = 0.18215;

  /**
   * 잠재 공간 텐서를 역스케일링합니다: z / scalingFactor
   */
  public static unscaleLatents(latents: Float32Array, scaleFactor: number = VAEDecoder.DEFAULT_SCALE_FACTOR): Float32Array {
    if (scaleFactor === 0) {
      throw new Error('[VAEDecoder] scaleFactor cannot be zero');
    }
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
    height: number
  ): Uint8ClampedArray {
    assertAllFinite('rgbTensor before RGBA conversion', rgbTensor);
    const totalPixels = width * height;
    assertLength('rgbTensor for RGBA conversion', rgbTensor.length, totalPixels * 3);

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
   * 가중치가 누락되었을 때 어떠한 가짜 가중치도 자동 생성하지 않고 즉각 예외를 분출합니다.
   */
  public static decode(
    latents: Float32Array,
    latentWidth: number,
    latentHeight: number,
    weights: VAEDecoderWeights,
    scaleFactor: number = VAEDecoder.DEFAULT_SCALE_FACTOR
  ): DecodedImage {
    // 1. 가중치 필수 검증 (침묵 폴백 원천 박멸)
    if (!weights) {
      throw new Error('[VAEDecoder] VAE decoder weights are required. Refusing to decode with synthetic weights.');
    }

    // 2. 입력 텐서 형상 및 유한성 검증
    assertLength('latents', latents.length, 4 * latentHeight * latentWidth);
    assertAllFinite('latents input', latents);

    const unscaled = this.unscaleLatents(latents, scaleFactor);
    assertAllFinite('unscaled latents', unscaled);

    const cMid = 32;

    // 3. 고정 계층 가중치 형상 사전 검증
    assertLength('postQuantConvWeight', weights.postQuantConvWeight.length, 4 * 4 * 1 * 1);
    if (weights.postQuantConvBias) {
      assertLength('postQuantConvBias', weights.postQuantConvBias.length, 4);
    }
    assertLength('convInWeight', weights.convInWeight.length, cMid * 4 * 3 * 3);
    if (weights.convInBias) {
      assertLength('convInBias', weights.convInBias.length, cMid);
    }
    assertLength('normOutGamma', weights.normOutGamma.length, cMid);
    assertLength('normOutBeta', weights.normOutBeta.length, cMid);
    assertLength('convOutWeight', weights.convOutWeight.length, 3 * cMid * 3 * 3);
    if (weights.convOutBias) {
      assertLength('convOutBias', weights.convOutBias.length, 3);
    }

    if (!weights.upBlocks || weights.upBlocks.length < 3) {
      throw new Error(`[VAEDecoder] VAE decoder requires exactly 3 upsample stages in upBlocks, received: ${weights.upBlocks?.length ?? 0}`);
    }

    // 4. Post-Quant Conv (1x1 Conv, 4 -> 4)
    const postQuant = this.conv2d(unscaled, 4, 4, latentHeight, latentWidth, weights.postQuantConvWeight, weights.postQuantConvBias, 1, 0);
    assertAllFinite('postQuant output', postQuant);

    // 5. Conv In (3x3 Conv, 4 -> 32)
    const featIn = this.conv2d(postQuant, 4, cMid, latentHeight, latentWidth, weights.convInWeight, weights.convInBias, 3, 1);
    assertAllFinite('featIn output', featIn);

    let currentFeat = featIn;
    let currentH = latentHeight;
    let currentW = latentWidth;
    let currentC = cMid;

    // 6. 3단계 업샘플링 계층: 64x64 -> 128x128 -> 256x256 -> 512x512
    for (let stage = 0; stage < 3; stage++) {
      const stageBlock = weights.upBlocks[stage];
      if (!stageBlock || !stageBlock.upsampleConvWeight) {
        throw new Error(`[VAEDecoder] Missing VAE upsample convolution weight at stage ${stage}`);
      }
      if (!stageBlock.normGamma || !stageBlock.normBeta) {
        throw new Error(`[VAEDecoder] Missing VAE upsample norm parameters at stage ${stage}`);
      }

      assertLength(`upBlocks[${stage}].upsampleConvWeight`, stageBlock.upsampleConvWeight.length, currentC * currentC * 3 * 3);
      if (stageBlock.upsampleConvBias) {
        assertLength(`upBlocks[${stage}].upsampleConvBias`, stageBlock.upsampleConvBias.length, currentC);
      }
      assertLength(`upBlocks[${stage}].normGamma`, stageBlock.normGamma.length, currentC);
      assertLength(`upBlocks[${stage}].normBeta`, stageBlock.normBeta.length, currentC);

      const nextH = currentH * 2;
      const nextW = currentW * 2;

      // Upsample2D (Bilinear 2x)
      const upsampled = this.upsample2d(currentFeat, currentC, currentH, currentW, nextH, nextW);
      assertAllFinite(`upsampled stage ${stage}`, upsampled);

      // Conv2d (3x3)
      const convOutStage = this.conv2d(upsampled, currentC, currentC, nextH, nextW, stageBlock.upsampleConvWeight, stageBlock.upsampleConvBias, 3, 1);
      assertAllFinite(`convOutStage stage ${stage}`, convOutStage);

      // GroupNorm (32 groups) + SiLU
      const normed = this.groupNorm(convOutStage, currentC, nextH, nextW, Math.min(32, currentC), stageBlock.normGamma, stageBlock.normBeta);
      assertAllFinite(`normed stage ${stage}`, normed);

      currentFeat = this.silu(normed);
      assertAllFinite(`silu stage ${stage}`, currentFeat);

      currentH = nextH;
      currentW = nextW;
    }

    // 7. Final Output Norm (GroupNorm 32 + SiLU)
    const finalNormed = this.groupNorm(currentFeat, currentC, currentH, currentW, Math.min(32, currentC), weights.normOutGamma, weights.normOutBeta);
    assertAllFinite('finalNormed', finalNormed);

    const finalAct = this.silu(finalNormed);
    assertAllFinite('finalAct', finalAct);

    // 8. Conv Out (3x3 Conv, currentC -> 3 채널 RGB)
    const rgbTensor = this.conv2d(finalAct, currentC, 3, currentH, currentW, weights.convOutWeight, weights.convOutBias, 3, 1);
    assertAllFinite('rgbTensor', rgbTensor);

    // 9. RGBA Canvas 변환
    const rgba = this.tensorToRGBA(rgbTensor, currentW, currentH);

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
    scaleFactor?: number
  ): DecodedImage {
    const expectedW = latentWidth * 8;
    const expectedH = latentHeight * 8;
    if (outWidth !== expectedW || outHeight !== expectedH) {
      throw new Error(`[VAEDecoder] decodeLatentToRGB scale mismatch: requested ${outWidth}x${outHeight}, but latent ${latentWidth}x${latentHeight} scales to ${expectedW}x${expectedH}`);
    }
    return this.decode(latents, latentWidth, latentHeight, weights, scaleFactor);
  }

  // --- 수치 신경망 기본 연산자 (엄격한 경계 조건 및 결함 검증 탑재) ---

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
    if (inC <= 0 || outC <= 0 || H <= 0 || W <= 0 || kernelSize <= 0) {
      throw new Error(`[VAEDecoder] conv2d invalid dimensions: inC=${inC}, outC=${outC}, H=${H}, W=${W}, kernelSize=${kernelSize}`);
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
    if (!Number.isInteger(G) || G <= 0) {
      throw new Error(`[VAEDecoder] GroupNorm group count must be positive integer: ${G}`);
    }
    if (C % G !== 0) {
      throw new Error(`[VAEDecoder] GroupNorm requires C divisible by G: C=${C}, G=${G}`);
    }
    if (gamma.length !== C || beta.length !== C) {
      throw new Error(`[VAEDecoder] GroupNorm affine parameter mismatch: C=${C}, gamma=${gamma.length}, beta=${beta.length}`);
    }
    assertLength('x in groupNorm', x.length, C * H * W);

    const hw = H * W;
    const channelsPerGroup = C / G;
    const groupSize = channelsPerGroup * hw;
    const out = new Float32Array(x.length);

    for (let g = 0; g < G; g++) {
      let sum = 0.0;
      let sqSum = 0.0;
      const baseC = g * channelsPerGroup;

      for (let c = 0; c < channelsPerGroup; c++) {
        const cIdx = (baseC + c) * hw;
        for (let i = 0; i < hw; i++) {
          const val = x[cIdx + i];
          sum += val;
          sqSum += val * val;
        }
      }

      const mean = sum / groupSize;
      const variance = Math.max(0.0, (sqSum / groupSize) - mean * mean);
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
    if (C <= 0 || H_in <= 0 || W_in <= 0 || H_out <= 0 || W_out <= 0) {
      throw new Error(`[VAEDecoder] upsample2d invalid dimensions: C=${C}, H_in=${H_in}, W_in=${W_in}, H_out=${H_out}, W_out=${W_out}`);
    }
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

/**
 * 테스트 스위트 전용 가중치 생성 유틸리티 (운영 코드에서는 호출 불가하도록 분리 격리)
 */
export class VAEDecoderTestFixtures {
  public static createSyntheticWeights(inC: number = 4, midC: number = 32): VAEDecoderWeights {
    const createWeights = (ic: number, oc: number, k: number) => {
      const len = oc * ic * k * k;
      const w = new Float32Array(len);
      const std = Math.sqrt(2.0 / (ic * k * k));
      for (let i = 0; i < len; i++) {
        w[i] = ((i % 100) / 100.0 - 0.5) * std;
      }
      return w;
    };

    return {
      postQuantConvWeight: createWeights(inC, inC, 1),
      postQuantConvBias: new Float32Array(inC).fill(0.0),
      convInWeight: createWeights(inC, midC, 3),
      convInBias: new Float32Array(midC).fill(0.0),
      normOutGamma: new Float32Array(midC).fill(1.0),
      normOutBeta: new Float32Array(midC).fill(0.0),
      convOutWeight: createWeights(midC, 3, 3),
      convOutBias: new Float32Array(3).fill(0.0),
      upBlocks: [
        {
          upsampleConvWeight: createWeights(midC, midC, 3),
          upsampleConvBias: new Float32Array(midC).fill(0.0),
          normGamma: new Float32Array(midC).fill(1.0),
          normBeta: new Float32Array(midC).fill(0.0),
        },
        {
          upsampleConvWeight: createWeights(midC, midC, 3),
          upsampleConvBias: new Float32Array(midC).fill(0.0),
          normGamma: new Float32Array(midC).fill(1.0),
          normBeta: new Float32Array(midC).fill(0.0),
        },
        {
          upsampleConvWeight: createWeights(midC, midC, 3),
          upsampleConvBias: new Float32Array(midC).fill(0.0),
          normGamma: new Float32Array(midC).fill(1.0),
          normBeta: new Float32Array(midC).fill(0.0),
        },
      ],
    };
  }
}
