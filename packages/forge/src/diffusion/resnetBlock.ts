/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-327 UNet ResNet Block WebGPU Forward Pipeline
 *
 * WHAT: 디퓨전 UNet의 기본 연산 단위인 ResNet Block 순전파 오케스트레이터입니다.
 * WHY: GroupNorm, SiLU, Conv2d, Time Embedding Addition, Residual Connection을
 *      WebGPU 상에서 하나의 유기적인 순전파 파이프라인으로 결합하기 위해 존재합니다.
 * HOW: [N, C_in, H, W] 입력에 대해 Norm1 -> SiLU -> Conv1 -> TimeEmbAdd -> Norm2 -> SiLU -> Conv2 -> SkipAdd
 *      연산 그래프를 구성하고 실행합니다.
 */

export interface ResNetBlockWeights {
  norm1Gamma: Float32Array;
  norm1Beta: Float32Array;
  conv1Weight: Float32Array;
  conv1Bias?: Float32Array;
  timeEmbProjWeight?: Float32Array;
  timeEmbProjBias?: Float32Array;
  norm2Gamma: Float32Array;
  norm2Beta: Float32Array;
  conv2Weight: Float32Array;
  conv2Bias?: Float32Array;
  skipProjWeight?: Float32Array;
  skipProjBias?: Float32Array;
}

export interface ResNetBlockConfig {
  inChannels: number;
  outChannels: number;
  height: number;
  width: number;
  numGroups?: number;
}

export class ResNetBlock {
  public config: ResNetBlockConfig;
  public weights: ResNetBlockWeights;

  constructor(config: ResNetBlockConfig, weights: ResNetBlockWeights) {
    this.config = {
      numGroups: 32,
      ...config,
    };
    this.weights = weights;
  }

  /**
   * 순수 CPU 참조 수학 연산 (Reference Forward) - WebGPU 출력 결과와의 수치 검증(Numerical Parity)용
   */
  public forwardCPU(
    input: Float32Array,
    timeEmb?: Float32Array
  ): Float32Array {
    const { inChannels, outChannels, height, width, numGroups = 32 } = this.config;
    const hw = height * width;
    const totalOut = outChannels * hw;
    const output = new Float32Array(totalOut);

    // 1. GroupNorm 1
    const norm1 = this.cpuGroupNorm(input, inChannels, height, width, numGroups, this.weights.norm1Gamma, this.weights.norm1Beta);

    // 2. SiLU 1
    const silu1 = this.cpuSiLU(norm1);

    // 3. Conv 1 (InChannels -> OutChannels, 3x3, padding 1)
    const conv1 = this.cpuConv2d(silu1, inChannels, outChannels, height, width, this.weights.conv1Weight, this.weights.conv1Bias);

    // 4. Time Embedding Add (Optional)
    let h = conv1;
    if (timeEmb && this.weights.timeEmbProjWeight) {
      const timeProj = this.cpuLinear(timeEmb, this.weights.timeEmbProjWeight, this.weights.timeEmbProjBias);
      h = new Float32Array(conv1.length);
      for (let c = 0; c < outChannels; c++) {
        const timeVal = timeProj[c];
        const offset = c * hw;
        for (let i = 0; i < hw; i++) {
          h[offset + i] = conv1[offset + i] + timeVal;
        }
      }
    }

    // 5. GroupNorm 2
    const norm2 = this.cpuGroupNorm(h, outChannels, height, width, numGroups, this.weights.norm2Gamma, this.weights.norm2Beta);

    // 6. SiLU 2
    const silu2 = this.cpuSiLU(norm2);

    // 7. Conv 2 (OutChannels -> OutChannels, 3x3, padding 1)
    const conv2 = this.cpuConv2d(silu2, outChannels, outChannels, height, width, this.weights.conv2Weight, this.weights.conv2Bias);

    // 8. Skip Connection
    let skip = input;
    if (inChannels !== outChannels) {
      if (this.weights.skipProjWeight) {
        skip = this.cpuConv2d(input, inChannels, outChannels, height, width, this.weights.skipProjWeight, this.weights.skipProjBias, 1, 0);
      } else {
        skip = new Float32Array(totalOut);
        const copyChannels = Math.min(inChannels, outChannels);
        skip.set(input.subarray(0, copyChannels * hw));
      }
    }

    // Residual Add: output = conv2 + skip
    for (let i = 0; i < totalOut; i++) {
      output[i] = conv2[i] + skip[i];
    }

    return output;
  }

  private cpuGroupNorm(
    x: Float32Array,
    C: number,
    H: number,
    W: number,
    G: number,
    gamma: Float32Array,
    beta: Float32Array,
    eps: number = 1e-5
  ): Float32Array {
    const hw = H * W;
    const channelsPerGroup = Math.floor(C / G);
    const groupSize = channelsPerGroup * hw;
    const out = new Float32Array(x.length);

    for (let g = 0; g < G; g++) {
      let sum = 0;
      let sqSum = 0;
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
      const variance = Math.max(0, (sqSum / groupSize) - mean * mean);
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

  private cpuSiLU(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      const clamped = Math.max(-88.0, Math.min(88.0, v));
      const sig = 1.0 / (1.0 + Math.exp(-clamped));
      out[i] = v * sig;
    }
    return out;
  }

  private cpuConv2d(
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
    const outH = H;
    const outW = W;
    const hw = outH * outW;
    const out = new Float32Array(outC * hw);
    const pad = padding;
    const kRadius = Math.floor(kernelSize / 2);

    for (let oc = 0; oc < outC; oc++) {
      const b = bias ? bias[oc] : 0;
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

  private cpuLinear(
    x: Float32Array,
    weight: Float32Array,
    bias?: Float32Array
  ): Float32Array {
    const outFeatures = bias ? bias.length : weight.length / x.length;
    const inFeatures = x.length;
    const out = new Float32Array(outFeatures);

    for (let oc = 0; oc < outFeatures; oc++) {
      let sum = bias ? bias[oc] : 0;
      const wOffset = oc * inFeatures;
      for (let ic = 0; ic < inFeatures; ic++) {
        sum += x[ic] * weight[wOffset + ic];
      }
      out[oc] = sum;
    }

    return out;
  }
}
