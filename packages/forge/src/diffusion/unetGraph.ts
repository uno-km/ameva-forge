/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-332 & SCRUM-335 UNet Denoising Neural Network Execution Graph
 *
 * WHAT: 시간 임베딩, 다운블록, 미드블록, 업블록 및 텍스트 교차 어텐션(Cross-Attention)을
 *      하나의 유기적인 순전파 신경망 그래프로 실행하는 UNet 엔진입니다.
 * WHY: 가짜 감쇠 수식을 영구 박멸하고, WebGPU WGSL Tiled GEMM 셰이더 기반 하드웨어 가속을 직결하기 위해 존재합니다.
 * HOW: Sinusoidal TimeEmbedding -> DownBlocks(ResNet + CrossAttn) -> MidBlock -> UpBlocks(Upsample + Skip Concat + ResNet + CrossAttn) -> OutConv.
 */

import { ResNetBlock, ResNetBlockWeights } from './resnetBlock';
import { VAEDecoder } from './vaeDecoder';
import { gpuCore, uploadFloat32Array, read, dispose } from '../tensor/gpuCore';
import { getDevice } from '../webgpu/device';
import { AMEVAForgeValidationError } from '../errors';

export interface SpatialCrossAttentionWeights {
  normGamma: Float32Array; // [latentC]
  normBeta: Float32Array;  // [latentC]
  qWeight: Float32Array;   // [latentC, latentC, 1, 1]
  qBias?: Float32Array;    // [latentC]
  kWeight: Float32Array;   // [latentC, textDim, 1, 1]
  kBias?: Float32Array;    // [latentC]
  vWeight: Float32Array;   // [latentC, textDim, 1, 1]
  vBias?: Float32Array;    // [latentC]
  outWeight: Float32Array; // [latentC, latentC, 1, 1]
  outBias?: Float32Array;  // [latentC]
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
  convInWeight: Float32Array;       // [C_base, 4, 3, 3]
  convInBias?: Float32Array;        // [C_base]
  timeMlp1Weight: Float32Array;     // [timeDim * 4, timeDim]
  timeMlp1Bias: Float32Array;       // [timeDim * 4]
  timeMlp2Weight: Float32Array;     // [timeDim * 4, timeDim * 4]
  timeMlp2Bias: Float32Array;       // [timeDim * 4]
  downBlocks: UNetBlockWeights[];
  midBlock: {
    resnet1: ResNetBlockWeights;
    attention: SpatialCrossAttentionWeights;
    resnet2: ResNetBlockWeights;
  };
  upBlocks: UNetBlockWeights[];
  normOutGamma: Float32Array;       // [C_base]
  normOutBeta: Float32Array;        // [C_base]
  convOutWeight: Float32Array;      // [4, C_base, 3, 3]
  convOutBias?: Float32Array;       // [4]
}

export class UNetGraph {
  public static readonly TIME_DIM = 320;

  /**
   * 정현파(Sinusoidal) 시간 임베딩 계산:
   * PE(t, 2i) = sin(t / 10000^(2i/d)), PE(t, 2i+1) = cos(t / 10000^(2i/d))
   */
  public static computeSinusoidalTimeEmbedding(timestep: number, dim: number = UNetGraph.TIME_DIM): Float32Array {
    const emb = new Float32Array(dim);
    const halfDim = Math.floor(dim / 2);
    const logFactor = Math.log(10000.0) / (halfDim - 1);

    for (let i = 0; i < halfDim; i++) {
      const freq = Math.exp(-i * logFactor);
      const arg = timestep * freq;
      emb[i] = Math.sin(arg);
      emb[halfDim + i] = Math.cos(arg);
    }
    return emb;
  }

  /**
   * Spatial Cross-Attention (CPU Reference):
   * Latent Q와 텍스트 임베딩 K, V 사이의 행렬 곱셈을 통한 의미론적 조건 주입
   */
  public static forwardCrossAttention(
    x: Float32Array,
    C: number,
    H: number,
    W: number,
    context: Float32Array, // [77, textDim]
    textSeqLen: number,
    textDim: number,
    weights: SpatialCrossAttentionWeights
  ): Float32Array {
    const hw = H * W;

    // 1. GroupNorm Latent
    const normed = VAEDecoder.groupNorm(x, C, H, W, Math.min(32, C), weights.normGamma, weights.normBeta);

    // 2. Q projection from Latent [C, hw]
    const q = VAEDecoder.conv2d(normed, C, C, H, W, weights.qWeight, weights.qBias, 1, 0);

    // 3. K, V projections from Text Context [77, textDim] -> [77, C]
    const k = new Float32Array(textSeqLen * C);
    const v = new Float32Array(textSeqLen * C);

    for (let t = 0; t < textSeqLen; t++) {
      const ctxOffset = t * textDim;
      const kvOffset = t * C;

      for (let oc = 0; oc < C; oc++) {
        let sumK = weights.kBias ? weights.kBias[oc] : 0.0;
        let sumV = weights.vBias ? weights.vBias[oc] : 0.0;
        const wOffset = oc * textDim;

        for (let ic = 0; ic < textDim; ic++) {
          const val = context[ctxOffset + ic];
          sumK += val * weights.kWeight[wOffset + ic];
          sumV += val * weights.vWeight[wOffset + ic];
        }
        k[kvOffset + oc] = sumK;
        v[kvOffset + oc] = sumV;
      }
    }

    // 4. Scaled Dot-Product Cross-Attention: Q [hw, C], K [77, C] -> Attn [hw, 77]
    const scale = 1.0 / Math.sqrt(C);
    const attended = new Float32Array(C * hw);

    for (let i = 0; i < hw; i++) {
      let maxScore = -Infinity;
      const scores = new Float32Array(textSeqLen);

      for (let t = 0; t < textSeqLen; t++) {
        let dot = 0.0;
        for (let c = 0; c < C; c++) {
          dot += q[c * hw + i] * k[t * C + c];
        }
        const s = dot * scale;
        scores[t] = s;
        if (s > maxScore) maxScore = s;
      }

      // Softmax
      let expSum = 0.0;
      for (let t = 0; t < textSeqLen; t++) {
        const e = Math.exp(scores[t] - maxScore);
        scores[t] = e;
        expSum += e;
      }
      const invSum = 1.0 / (expSum + 1e-9);
      for (let t = 0; t < textSeqLen; t++) {
        scores[t] *= invSum;
      }

      // Context multiplication: V [77, C]
      for (let c = 0; c < C; c++) {
        let cVal = 0.0;
        for (let t = 0; t < textSeqLen; t++) {
          cVal += v[t * C + c] * scores[t];
        }
        attended[c * hw + i] = cVal;
      }
    }

    // 5. Out 1x1 Conv projection
    const outProj = VAEDecoder.conv2d(attended, C, C, H, W, weights.outWeight, weights.outBias, 1, 0);

    // 6. Residual Skip
    const res = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      res[i] = x[i] + outProj[i];
    }
    return res;
  }

  /**
   * Spatial Cross-Attention (WebGPU Hardware Accelerated):
   * WebGPU Tiled GEMM 셰이더를 통해 K, V 사상 및 QK^T 어텐션 연산을 하드웨어 가속합니다.
   */
  public static async forwardCrossAttentionGPU(
    x: Float32Array,
    C: number,
    H: number,
    W: number,
    context: Float32Array,
    textSeqLen: number,
    textDim: number,
    weights: SpatialCrossAttentionWeights
  ): Promise<Float32Array> {
    const hw = H * W;
    const dev = getDevice();
    if (!dev) {
      throw new AMEVAForgeValidationError('[UNetGraph:WebGPU] WebGPU device is not initialized. Cannot run forwardCrossAttentionGPU.');
    }

    // 1. GroupNorm Latent
    const normed = VAEDecoder.groupNorm(x, C, H, W, Math.min(32, C), weights.normGamma, weights.normBeta);

    // 2. Q projection from Latent [C, hw] -> [hw, C]
    const qRaw = VAEDecoder.conv2d(normed, C, C, H, W, weights.qWeight, weights.qBias, 1, 0);
    const qTransposed = new Float32Array(hw * C);
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < hw; i++) {
        qTransposed[i * C + c] = qRaw[c * hw + i];
      }
    }

    // 3. Upload to GPU
    const hQ = uploadFloat32Array(qTransposed, [hw, C]);
    const hCtx = uploadFloat32Array(context, [textSeqLen, textDim]);
    const hKw = uploadFloat32Array(weights.kWeight, [C, textDim]);
    const hVw = uploadFloat32Array(weights.vWeight, [C, textDim]);
    const hOutW = uploadFloat32Array(weights.outWeight, [C, C]);

    const handlesToDispose = [hQ, hCtx, hKw, hVw, hOutW];

    try {
      // K = Context [textSeqLen, textDim] @ Kw^T [textDim, C] -> [textSeqLen, C]
      const hKwT = gpuCore.transpose(hKw);
      handlesToDispose.push(hKwT);
      const hK = gpuCore.matmul(hCtx, hKwT);
      handlesToDispose.push(hK);

      // V = Context [textSeqLen, textDim] @ Vw^T [textDim, C] -> [textSeqLen, C]
      const hVwT = gpuCore.transpose(hVw);
      handlesToDispose.push(hVwT);
      const hV = gpuCore.matmul(hCtx, hVwT);
      handlesToDispose.push(hV);

      // Scaled Dot-Product: Q [hw, C] @ K^T [C, textSeqLen] -> [hw, textSeqLen]
      const hKT = gpuCore.transpose(hK);
      handlesToDispose.push(hKT);
      const hScores = gpuCore.matmul(hQ, hKT);
      handlesToDispose.push(hScores);

      // Read back raw scores for Softmax & scaling
      const rawScores = await read(hScores);
      const scale = 1.0 / Math.sqrt(C);
      const softmaxScores = new Float32Array(hw * textSeqLen);

      for (let i = 0; i < hw; i++) {
        const off = i * textSeqLen;
        let maxS = -Infinity;
        for (let t = 0; t < textSeqLen; t++) {
          const s = rawScores[off + t] * scale;
          softmaxScores[off + t] = s;
          if (s > maxS) maxS = s;
        }
        let sumExp = 0.0;
        for (let t = 0; t < textSeqLen; t++) {
          const e = Math.exp(softmaxScores[off + t] - maxS);
          softmaxScores[off + t] = e;
          sumExp += e;
        }
        const invSum = 1.0 / (sumExp + 1e-9);
        for (let t = 0; t < textSeqLen; t++) {
          softmaxScores[off + t] *= invSum;
        }
      }

      // Context multiplication on GPU: Attended [hw, textSeqLen] @ V [textSeqLen, C] -> [hw, C]
      const hSoftmax = uploadFloat32Array(softmaxScores, [hw, textSeqLen]);
      handlesToDispose.push(hSoftmax);
      const hAttended = gpuCore.matmul(hSoftmax, hV);
      handlesToDispose.push(hAttended);

      // Out 1x1 projection on GPU: Attended [hw, C] @ OutW^T [C, C] -> [hw, C]
      const hOutWT = gpuCore.transpose(hOutW);
      handlesToDispose.push(hOutWT);
      const hOutProj = gpuCore.matmul(hAttended, hOutWT);
      handlesToDispose.push(hOutProj);

      const outProjFlat = await read(hOutProj);

      // Add residual to x
      const res = new Float32Array(x.length);
      for (let c = 0; c < C; c++) {
        const cBias = weights.outBias ? weights.outBias[c] : 0.0;
        for (let i = 0; i < hw; i++) {
          const outVal = outProjFlat[i * C + c] + cBias;
          res[c * hw + i] = x[c * hw + i] + outVal;
        }
      }

      return res;
    } finally {
      for (const h of handlesToDispose) {
        try { dispose(h); } catch {}
      }
    }
  }

  /**
   * UNet 디노이징 신경망 전체 순전파 (CPU Reference):
   * 잠재 텐서(z_t) + 타임스텝(t) + 텍스트 컨텍스트 임베딩(c) -> 예측 노이즈(eps_theta)
   */
  public static forward(
    sample: Float32Array,
    timestep: number,
    textContext: Float32Array, // [77, 768]
    weights: UNetWeights,
    height: number = 64,
    width: number = 64,
    baseChannels: number = 32
  ): Float32Array {
    const hw = height * width;
    if (sample.length !== 4 * hw) {
      throw new Error(`[UNetGraph] sample length mismatch: expected ${4 * hw}, received ${sample.length}`);
    }

    // 1. Time Embedding MLP
    const rawTimeEmb = this.computeSinusoidalTimeEmbedding(timestep, UNetGraph.TIME_DIM);
    const timeMlpDim = UNetGraph.TIME_DIM * 4;

    // Linear 1
    const timeH1 = new Float32Array(timeMlpDim);
    for (let oc = 0; oc < timeMlpDim; oc++) {
      let sum = weights.timeMlp1Bias[oc];
      const wOff = oc * UNetGraph.TIME_DIM;
      for (let ic = 0; ic < UNetGraph.TIME_DIM; ic++) {
        sum += rawTimeEmb[ic] * weights.timeMlp1Weight[wOff + ic];
      }
      timeH1[oc] = sum;
    }
    const timeAct1 = VAEDecoder.silu(timeH1);

    // Linear 2
    const timeEmb = new Float32Array(timeMlpDim);
    for (let oc = 0; oc < timeMlpDim; oc++) {
      let sum = weights.timeMlp2Bias[oc];
      const wOff = oc * timeMlpDim;
      for (let ic = 0; ic < timeMlpDim; ic++) {
        sum += timeAct1[ic] * weights.timeMlp2Weight[wOff + ic];
      }
      timeEmb[oc] = sum;
    }

    // 2. Conv In (4 -> baseChannels, 3x3, pad 1)
    let h = VAEDecoder.conv2d(sample, 4, baseChannels, height, width, weights.convInWeight, weights.convInBias, 3, 1);
    const skipConnections: Float32Array[] = [];
    skipConnections.push(h);

    // 3. Down Blocks (ResNet + CrossAttn)
    for (let i = 0; i < weights.downBlocks.length; i++) {
      const block = weights.downBlocks[i];
      for (let r = 0; r < block.resnets.length; r++) {
        const resnet = new ResNetBlock(
          { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
          block.resnets[r]
        );
        h = resnet.forwardCPU(h, timeEmb);
        skipConnections.push(h);
      }
      for (let a = 0; a < block.attentions.length; a++) {
        h = this.forwardCrossAttention(h, baseChannels, height, width, textContext, 77, 768, block.attentions[a]);
        skipConnections.push(h);
      }
    }

    // 4. Mid Block (ResNet -> CrossAttn -> ResNet)
    const midRes1 = new ResNetBlock(
      { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
      weights.midBlock.resnet1
    );
    h = midRes1.forwardCPU(h, timeEmb);
    h = this.forwardCrossAttention(h, baseChannels, height, width, textContext, 77, 768, weights.midBlock.attention);

    const midRes2 = new ResNetBlock(
      { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
      weights.midBlock.resnet2
    );
    h = midRes2.forwardCPU(h, timeEmb);

    // 5. Up Blocks (Skip Connection Add + ResNet + CrossAttn)
    for (let i = 0; i < weights.upBlocks.length; i++) {
      const block = weights.upBlocks[i];
      for (let r = 0; r < block.resnets.length; r++) {
        const skip = skipConnections.pop() || h;
        for (let idx = 0; idx < h.length; idx++) {
          h[idx] += skip[idx];
        }
        const resnet = new ResNetBlock(
          { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
          block.resnets[r]
        );
        h = resnet.forwardCPU(h, timeEmb);
      }
      for (let a = 0; a < block.attentions.length; a++) {
        h = this.forwardCrossAttention(h, baseChannels, height, width, textContext, 77, 768, block.attentions[a]);
      }
    }

    // 6. Norm Out (GroupNorm + SiLU)
    const normedOut = VAEDecoder.groupNorm(h, baseChannels, height, width, Math.min(32, baseChannels), weights.normOutGamma, weights.normOutBeta);
    const actOut = VAEDecoder.silu(normedOut);

    // 7. Conv Out (baseChannels -> 4 channels predicted noise)
    const predNoise = VAEDecoder.conv2d(actOut, baseChannels, 4, height, width, weights.convOutWeight, weights.convOutBias, 3, 1);

    return predNoise;
  }

  /**
   * UNet 디노이징 신경망 전체 순전파 (WebGPU Hardware Accelerated):
   * WebGPU 장치 상에서 Tiled GEMM 기반 Cross-Attention을 수행하여 고해상도 지연시간을 단축합니다.
   */
  public static async forwardGPU(
    sample: Float32Array,
    timestep: number,
    textContext: Float32Array, // [77, 768]
    weights: UNetWeights,
    height: number = 64,
    width: number = 64,
    baseChannels: number = 32
  ): Promise<Float32Array> {
    const hw = height * width;
    if (sample.length !== 4 * hw) {
      throw new AMEVAForgeValidationError(`[UNetGraph:WebGPU] sample length mismatch: expected ${4 * hw}, received ${sample.length}`);
    }
    const dev = getDevice();
    if (!dev) {
      throw new AMEVAForgeValidationError('[UNetGraph:WebGPU] WebGPU device is not available. Refusing silent fallback to CPU.');
    }

    // 1. Time Embedding MLP
    const rawTimeEmb = this.computeSinusoidalTimeEmbedding(timestep, UNetGraph.TIME_DIM);
    const timeMlpDim = UNetGraph.TIME_DIM * 4;

    const timeH1 = new Float32Array(timeMlpDim);
    for (let oc = 0; oc < timeMlpDim; oc++) {
      let sum = weights.timeMlp1Bias[oc];
      const wOff = oc * UNetGraph.TIME_DIM;
      for (let ic = 0; ic < UNetGraph.TIME_DIM; ic++) {
        sum += rawTimeEmb[ic] * weights.timeMlp1Weight[wOff + ic];
      }
      timeH1[oc] = sum;
    }
    const timeAct1 = VAEDecoder.silu(timeH1);

    const timeEmb = new Float32Array(timeMlpDim);
    for (let oc = 0; oc < timeMlpDim; oc++) {
      let sum = weights.timeMlp2Bias[oc];
      const wOff = oc * timeMlpDim;
      for (let ic = 0; ic < timeMlpDim; ic++) {
        sum += timeAct1[ic] * weights.timeMlp2Weight[wOff + ic];
      }
      timeEmb[oc] = sum;
    }

    // 2. Conv In (4 -> baseChannels, 3x3, pad 1)
    let h = VAEDecoder.conv2d(sample, 4, baseChannels, height, width, weights.convInWeight, weights.convInBias, 3, 1);
    const skipConnections: Float32Array[] = [];
    skipConnections.push(h);

    // 3. Down Blocks (ResNet + CrossAttn GPU)
    for (let i = 0; i < weights.downBlocks.length; i++) {
      const block = weights.downBlocks[i];
      for (let r = 0; r < block.resnets.length; r++) {
        const resnet = new ResNetBlock(
          { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
          block.resnets[r]
        );
        h = resnet.forwardCPU(h, timeEmb);
        skipConnections.push(h);
      }
      for (let a = 0; a < block.attentions.length; a++) {
        h = await this.forwardCrossAttentionGPU(h, baseChannels, height, width, textContext, 77, 768, block.attentions[a]);
        skipConnections.push(h);
      }
    }

    // 4. Mid Block (ResNet -> CrossAttn GPU -> ResNet)
    const midRes1 = new ResNetBlock(
      { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
      weights.midBlock.resnet1
    );
    h = midRes1.forwardCPU(h, timeEmb);
    h = await this.forwardCrossAttentionGPU(h, baseChannels, height, width, textContext, 77, 768, weights.midBlock.attention);

    const midRes2 = new ResNetBlock(
      { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
      weights.midBlock.resnet2
    );
    h = midRes2.forwardCPU(h, timeEmb);

    // 5. Up Blocks (Skip Connection Add + ResNet + CrossAttn GPU)
    for (let i = 0; i < weights.upBlocks.length; i++) {
      const block = weights.upBlocks[i];
      for (let r = 0; r < block.resnets.length; r++) {
        const skip = skipConnections.pop() || h;
        for (let idx = 0; idx < h.length; idx++) {
          h[idx] += skip[idx];
        }
        const resnet = new ResNetBlock(
          { inChannels: baseChannels, outChannels: baseChannels, height, width, numGroups: Math.min(32, baseChannels) },
          block.resnets[r]
        );
        h = resnet.forwardCPU(h, timeEmb);
      }
      for (let a = 0; a < block.attentions.length; a++) {
        h = await this.forwardCrossAttentionGPU(h, baseChannels, height, width, textContext, 77, 768, block.attentions[a]);
      }
    }

    // 6. Norm Out (GroupNorm + SiLU)
    const normedOut = VAEDecoder.groupNorm(h, baseChannels, height, width, Math.min(32, baseChannels), weights.normOutGamma, weights.normOutBeta);
    const actOut = VAEDecoder.silu(normedOut);

    // 7. Conv Out (baseChannels -> 4 channels predicted noise)
    const predNoise = VAEDecoder.conv2d(actOut, baseChannels, 4, height, width, weights.convOutWeight, weights.convOutBias, 3, 1);

    return predNoise;
  }
}
