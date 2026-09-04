/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: CLIP ViT-B/16 Vision Transformer Forward Engine
 *
 * WHAT: 이미지를 16x16 패치로 분할하고 트랜스포머 인코더를 거쳐 768차원 시맨틱 특징 벡터를 추출하는 비전 인코더입니다.
 * WHY: 제로샷 이미지 분류, 텍스트-이미지 시맨틱 검색, VLM 멀티모달 시각 입력의 핵심 관문으로 동작합니다.
 * HOW: Patchify Conv2d -> Class Token Concat -> Position Embedding -> 12-Layer Vision Transformer -> LayerNorm.
 */

import { CLIPLayerWeights } from '../diffusion/clipTextEncoder';
import { VisionError, VisionErrorCode } from './classicalCV';
import { gpuCore, uploadFloat32Array, read, dispose } from '../tensor/gpuCore';
import { getDevice } from '../webgpu/device';


export interface CLIPVisionWeights {
  patchConvWeight: Float32Array;      // [768, 3, 16, 16]
  patchConvBias?: Float32Array;       // [768]
  classEmbedding: Float32Array;       // [768]
  positionEmbedding: Float32Array;    // [numPatches + 1, 768]
  preNormGamma: Float32Array;         // [768]
  preNormBeta: Float32Array;          // [768]
  layers: CLIPLayerWeights[];         // 12 Transformer layers
  postNormGamma: Float32Array;        // [768]
  postNormBeta: Float32Array;         // [768]
  projectionWeight?: Float32Array;    // [512, 768] (Optional CLIP output projection)
}

export class CLIPVisionEncoder {
  public static readonly PATCH_SIZE = 16;
  public static readonly EMBED_DIM = 768;
  public static readonly NUM_HEADS = 12;

  /**
   * RGB 이미지(3, H, W)를 16x16 패치로 분할하고 선형 투영합니다.
   */
  public static patchProjection(
    rgb: Float32Array,
    width: number,
    height: number,
    weights: Float32Array, // [768, 3, 16, 16]
    bias?: Float32Array
  ): { patches: Float32Array; numPatches: number } {
    const p = CLIPVisionEncoder.PATCH_SIZE;
    if (width % p !== 0 || height % p !== 0) {
      throw new VisionError(
        VisionErrorCode.INVALID_IMAGE_DIMENSIONS,
        `Image dimensions must be divisible by patch size (${p}), received: ${width}x${height}`
      );
    }

    const gridH = Math.floor(height / p);
    const gridW = Math.floor(width / p);
    const numPatches = gridH * gridW;
    const dim = CLIPVisionEncoder.EMBED_DIM;
    const out = new Float32Array(numPatches * dim);

    for (let gh = 0; gh < gridH; gh++) {
      for (let gw = 0; gw < gridW; gw++) {
        const patchIdx = gh * gridW + gw;
        const outOffset = patchIdx * dim;

        for (let oc = 0; oc < dim; oc++) {
          let sum = bias ? bias[oc] : 0.0;
          const wBase = oc * (3 * p * p);

          for (let c = 0; c < 3; c++) {
            const inCBase = c * (height * width);
            const wCBase = wBase + c * (p * p);

            for (let py = 0; py < p; py++) {
              const ih = gh * p + py;
              for (let px = 0; px < p; px++) {
                const iw = gw * p + px;
                const pixel = rgb[inCBase + ih * width + iw];
                const weight = weights[wCBase + py * p + px];
                sum += pixel * weight;
              }
            }
          }
          out[outOffset + oc] = sum;
        }
      }
    }

    return { patches: out, numPatches };
  }

  /**
   * CLIP Vision Transformer 전체 순전파:
   * 이미지 RGB -> 패치 임베딩 -> [CLS] 토큰 결합 -> 트랜스포머 레이어 -> 768차원 이미지 특징 벡터
   */
  public static forward(
    rgb: Float32Array,
    width: number,
    height: number,
    weights: CLIPVisionWeights
  ): { imageEmbedding: Float32Array; patchEmbeddings: Float32Array } {
    const dim = CLIPVisionEncoder.EMBED_DIM;
    const { patches, numPatches } = this.patchProjection(rgb, width, height, weights.patchConvWeight, weights.patchConvBias);

    // Sequence length: [CLS] token + image patches
    const seqLen = numPatches + 1;
    const tokens = new Float32Array(seqLen * dim);

    // 1. [CLS] Token 배치
    tokens.set(weights.classEmbedding, 0);

    // 2. 패치 토큰 배치
    tokens.set(patches, dim);

    // 3. Positional Embedding 결합
    for (let i = 0; i < seqLen; i++) {
      const off = i * dim;
      for (let d = 0; d < dim; d++) {
        tokens[off + d] += weights.positionEmbedding[off + d];
      }
    }

    // 4. Pre-LayerNorm
    let h = this.layerNorm(tokens, seqLen, dim, weights.preNormGamma, weights.preNormBeta);

    // 5. 12 Transformer Layers
    for (let l = 0; l < weights.layers.length; l++) {
      const layer = weights.layers[l];

      // Self-Attention
      const norm1 = this.layerNorm(h, seqLen, dim, layer.norm1Gamma, layer.norm1Beta);
      const attnOut = this.forwardSelfAttention(norm1, seqLen, dim, CLIPVisionEncoder.NUM_HEADS, layer);

      for (let i = 0; i < h.length; i++) {
        h[i] += attnOut[i];
      }

      // MLP
      const norm2 = this.layerNorm(h, seqLen, dim, layer.norm2Gamma, layer.norm2Beta);
      const mlpFc1 = this.linear(norm2, seqLen, dim, 3072, layer.mlpFc1Weight, layer.mlpFc1Bias);
      const gelu = this.quickGELU(mlpFc1);
      const mlpFc2 = this.linear(gelu, seqLen, 3072, dim, layer.mlpFc2Weight, layer.mlpFc2Bias);

      for (let i = 0; i < h.length; i++) {
        h[i] += mlpFc2[i];
      }
    }

    // 6. Post-LayerNorm
    const finalTokens = this.layerNorm(h, seqLen, dim, weights.postNormGamma, weights.postNormBeta);

    // 7. [CLS] 임베딩 추출 (글로벌 이미지 특징)
    const clsEmbedding = new Float32Array(dim);
    clsEmbedding.set(finalTokens.subarray(0, dim));

    // Optional: L2 정규화 (코사인 유사도 검색용)
    let normSq = 0.0;
    for (let d = 0; d < dim; d++) normSq += clsEmbedding[d] * clsEmbedding[d];
    const invNorm = 1.0 / (Math.sqrt(normSq) + 1e-9);
    for (let d = 0; d < dim; d++) clsEmbedding[d] *= invNorm;

    const patchTokens = new Float32Array(numPatches * dim);
    patchTokens.set(finalTokens.subarray(dim));

    return {
      imageEmbedding: clsEmbedding,
      patchEmbeddings: patchTokens,
    };
  }

  /**
   * CLIP Vision Transformer WebGPU 하드웨어 가속 순전파
   */
  public static async forwardGPU(
    rgb: Float32Array,
    width: number,
    height: number,
    weights: CLIPVisionWeights
  ): Promise<{ imageEmbedding: Float32Array; patchEmbeddings: Float32Array }> {
    const dev = getDevice();
    if (!dev) {
      throw new VisionError(VisionErrorCode.WEBGPU_NOT_AVAILABLE, '[CLIPVisionEncoder:WebGPU] WebGPU device is not available. Refusing silent fallback to CPU.');
    }
    const dim = CLIPVisionEncoder.EMBED_DIM;
    const { patches, numPatches } = this.patchProjection(rgb, width, height, weights.patchConvWeight, weights.patchConvBias);

    const seqLen = numPatches + 1;
    const tokens = new Float32Array(seqLen * dim);
    tokens.set(weights.classEmbedding, 0);
    tokens.set(patches, dim);

    for (let i = 0; i < seqLen; i++) {
      const off = i * dim;
      for (let d = 0; d < dim; d++) {
        tokens[off + d] += weights.positionEmbedding[off + d];
      }
    }

    let h = this.layerNorm(tokens, seqLen, dim, weights.preNormGamma, weights.preNormBeta);

    for (let l = 0; l < weights.layers.length; l++) {
      const layer = weights.layers[l];
      const norm1 = this.layerNorm(h, seqLen, dim, layer.norm1Gamma, layer.norm1Beta);

      const hNorm1 = uploadFloat32Array(norm1, [seqLen, dim]);
      const hWQ = uploadFloat32Array(layer.qProjWeight, [dim, dim]);
      const hWK = uploadFloat32Array(layer.kProjWeight, [dim, dim]);
      const hWV = uploadFloat32Array(layer.vProjWeight, [dim, dim]);
      const hWOut = uploadFloat32Array(layer.outProjWeight, [dim, dim]);
      const handles = [hNorm1, hWQ, hWK, hWV, hWOut];

      try {
        const hWQT = gpuCore.transpose(hWQ);
        const hWKT = gpuCore.transpose(hWK);
        const hWVT = gpuCore.transpose(hWV);
        handles.push(hWQT, hWKT, hWVT);

        const hQ = gpuCore.matmul(hNorm1, hWQT);
        const hK = gpuCore.matmul(hNorm1, hWKT);
        const hV = gpuCore.matmul(hNorm1, hWVT);
        handles.push(hQ, hK, hV);

        const q = await read(hQ);
        const k = await read(hK);
        const v = await read(hV);

        const headDim = Math.floor(dim / CLIPVisionEncoder.NUM_HEADS);
        const scale = 1.0 / Math.sqrt(headDim);
        const attnRaw = new Float32Array(seqLen * dim);

        for (let head = 0; head < CLIPVisionEncoder.NUM_HEADS; head++) {
          const headOff = head * headDim;
          for (let i = 0; i < seqLen; i++) {
            const qOff = i * dim + headOff;
            let maxScore = -Infinity;
            const scores = new Float32Array(seqLen);

            for (let j = 0; j < seqLen; j++) {
              const kOff = j * dim + headOff;
              let dot = 0.0;
              for (let d = 0; d < headDim; d++) {
                dot += q[qOff + d] * k[kOff + d];
              }
              const s = dot * scale;
              scores[j] = s;
              if (s > maxScore) maxScore = s;
            }

            let expSum = 0.0;
            for (let j = 0; j < seqLen; j++) {
              const e = Math.exp(scores[j] - maxScore);
              scores[j] = e;
              expSum += e;
            }
            const invSum = 1.0 / (expSum + 1e-9);
            for (let j = 0; j < seqLen; j++) scores[j] *= invSum;

            const outOff = i * dim + headOff;
            for (let d = 0; d < headDim; d++) {
              let val = 0.0;
              for (let j = 0; j < seqLen; j++) {
                val += scores[j] * v[j * dim + headOff + d];
              }
              attnRaw[outOff + d] = val;
            }
          }
        }

        const hAttnRaw = uploadFloat32Array(attnRaw, [seqLen, dim]);
        handles.push(hAttnRaw);
        const hWOutT = gpuCore.transpose(hWOut);
        handles.push(hWOutT);
        const hAttnOut = gpuCore.matmul(hAttnRaw, hWOutT);
        handles.push(hAttnOut);
        const attnOut = await read(hAttnOut);

        for (let i = 0; i < h.length; i++) {
          h[i] += attnOut[i];
        }

        const norm2 = this.layerNorm(h, seqLen, dim, layer.norm2Gamma, layer.norm2Beta);
        const hNorm2 = uploadFloat32Array(norm2, [seqLen, dim]);
        const hWFc1 = uploadFloat32Array(layer.mlpFc1Weight, [3072, dim]);
        const hWFc2 = uploadFloat32Array(layer.mlpFc2Weight, [dim, 3072]);
        handles.push(hNorm2, hWFc1, hWFc2);

        const hWFc1T = gpuCore.transpose(hWFc1);
        handles.push(hWFc1T);
        const hMlp1 = gpuCore.matmul(hNorm2, hWFc1T);
        handles.push(hMlp1);

        const mlp1Raw = await read(hMlp1);
        const gelu = this.quickGELU(mlp1Raw);

        const hGelu = uploadFloat32Array(gelu, [seqLen, 3072]);
        handles.push(hGelu);
        const hWFc2T = gpuCore.transpose(hWFc2);
        handles.push(hWFc2T);
        const hMlp2 = gpuCore.matmul(hGelu, hWFc2T);
        handles.push(hMlp2);

        const mlp2Raw = await read(hMlp2);
        for (let i = 0; i < h.length; i++) {
          h[i] += mlp2Raw[i];
        }
      } finally {
        for (const hnd of handles) {
          try { dispose(hnd); } catch {}
        }
      }
    }

    const finalTokens = this.layerNorm(h, seqLen, dim, weights.postNormGamma, weights.postNormBeta);

    const clsEmbedding = new Float32Array(dim);
    clsEmbedding.set(finalTokens.subarray(0, dim));

    let normSq = 0.0;
    for (let d = 0; d < dim; d++) normSq += clsEmbedding[d] * clsEmbedding[d];
    const invNorm = 1.0 / (Math.sqrt(normSq) + 1e-9);
    for (let d = 0; d < dim; d++) clsEmbedding[d] *= invNorm;

    const patchTokens = new Float32Array(numPatches * dim);
    patchTokens.set(finalTokens.subarray(dim));

    return {
      imageEmbedding: clsEmbedding,
      patchEmbeddings: patchTokens,
    };
  }

  // --- 보조 수학 연산 ---
  private static layerNorm(x: Float32Array, seqLen: number, dim: number, gamma: Float32Array, beta: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < seqLen; i++) {
      const off = i * dim;
      let sum = 0.0;
      for (let d = 0; d < dim; d++) sum += x[off + d];
      const mean = sum / dim;

      let sqDiff = 0.0;
      for (let d = 0; d < dim; d++) {
        const diff = x[off + d] - mean;
        sqDiff += diff * diff;
      }
      const invStd = 1.0 / Math.sqrt(sqDiff / dim + 1e-5);
      for (let d = 0; d < dim; d++) {
        out[off + d] = (x[off + d] - mean) * invStd * gamma[d] + beta[d];
      }
    }
    return out;
  }

  private static quickGELU(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      const clamped = Math.max(-88.0, Math.min(88.0, 1.702 * v));
      out[i] = v * (1.0 / (1.0 + Math.exp(-clamped)));
    }
    return out;
  }

  private static linear(x: Float32Array, seqLen: number, inDim: number, outDim: number, w: Float32Array, b?: Float32Array): Float32Array {
    const out = new Float32Array(seqLen * outDim);
    for (let i = 0; i < seqLen; i++) {
      const inOff = i * inDim;
      const outOff = i * outDim;
      for (let oc = 0; oc < outDim; oc++) {
        let sum = b ? b[oc] : 0.0;
        const wOff = oc * inDim;
        for (let ic = 0; ic < inDim; ic++) {
          sum += x[inOff + ic] * w[wOff + ic];
        }
        out[outOff + oc] = sum;
      }
    }
    return out;
  }

  private static forwardSelfAttention(x: Float32Array, seqLen: number, dim: number, numHeads: number, layer: CLIPLayerWeights): Float32Array {
    const headDim = Math.floor(dim / numHeads);
    const scale = 1.0 / Math.sqrt(headDim);

    const q = this.linear(x, seqLen, dim, dim, layer.qProjWeight, layer.qProjBias);
    const k = this.linear(x, seqLen, dim, dim, layer.kProjWeight, layer.kProjBias);
    const v = this.linear(x, seqLen, dim, dim, layer.vProjWeight, layer.vProjBias);

    const out = new Float32Array(seqLen * dim);

    for (let h = 0; h < numHeads; h++) {
      const headOff = h * headDim;

      for (let i = 0; i < seqLen; i++) {
        const qOff = i * dim + headOff;
        let maxScore = -Infinity;
        const scores = new Float32Array(seqLen);

        for (let j = 0; j < seqLen; j++) {
          const kOff = j * dim + headOff;
          let dot = 0.0;
          for (let d = 0; d < headDim; d++) {
            dot += q[qOff + d] * k[kOff + d];
          }
          const s = dot * scale;
          scores[j] = s;
          if (s > maxScore) maxScore = s;
        }

        let expSum = 0.0;
        for (let j = 0; j < seqLen; j++) {
          const e = Math.exp(scores[j] - maxScore);
          scores[j] = e;
          expSum += e;
        }
        const invSum = 1.0 / (expSum + 1e-9);
        for (let j = 0; j < seqLen; j++) scores[j] *= invSum;

        const outOff = i * dim + headOff;
        for (let d = 0; d < headDim; d++) {
          let val = 0.0;
          for (let j = 0; j < seqLen; j++) {
            val += scores[j] * v[j * dim + headOff + d];
          }
          out[outOff + d] = val;
        }
      }
    }

    return this.linear(out, seqLen, dim, dim, layer.outProjWeight, layer.outProjBias);
  }
}
