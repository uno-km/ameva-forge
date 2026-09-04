/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-334 & SCRUM-335 Multimodal VLM Projector Engine
 *
 * WHAT: 비전 패치 임베딩 [N, 768]을 언어 모델(LLM) 텍스트 임베딩 공간 [N, textDim]으로 매핑하는 멀티모달 프로젝터입니다.
 * WHY: 이미지를 본 후 LLM이 그 내용을 텍스트로 추론하여 자연어로 답변할 수 있도록 시각-언어 공간을 정렬하고,
 *      WebGPU WGSL Tiled GEMM 셰이더를 통해 VRAM 내에서 하드웨어 가속 사상합니다.
 */

import { gpuCore, uploadFloat32Array, read, dispose } from '../tensor/gpuCore';
import { getDevice } from '../webgpu/device';
import { AMEVAForgeValidationError } from '../errors';

export interface VLMProjectorWeights {
  mlp1Weight: Float32Array; // [hiddenDim, 768]
  mlp1Bias?: Float32Array;  // [hiddenDim]
  mlp2Weight: Float32Array; // [llmDim, hiddenDim]
  mlp2Bias?: Float32Array;  // [llmDim]
}

export class VLMProjector {
  /**
   * 2-Layer GeLU MLP 프로젝터 순전파 (CPU Reference)
   */
  public static project(
    visualTokens: Float32Array,
    numTokens: number,
    weights: VLMProjectorWeights,
    hiddenDim: number = 2048,
    llmDim: number = 2048
  ): Float32Array {
    const inDim = 768;
    const h1 = new Float32Array(numTokens * hiddenDim);

    // Linear 1
    for (let t = 0; t < numTokens; t++) {
      const inOff = t * inDim;
      const hOff = t * hiddenDim;
      for (let oc = 0; oc < hiddenDim; oc++) {
        let sum = weights.mlp1Bias ? weights.mlp1Bias[oc] : 0.0;
        const wOff = oc * inDim;
        for (let ic = 0; ic < inDim; ic++) {
          sum += visualTokens[inOff + ic] * weights.mlp1Weight[wOff + ic];
        }
        // GeLU
        const clamped = Math.max(-88.0, Math.min(88.0, 1.702 * sum));
        h1[hOff + oc] = sum * (1.0 / (1.0 + Math.exp(-clamped)));
      }
    }

    // Linear 2
    const projected = new Float32Array(numTokens * llmDim);
    for (let t = 0; t < numTokens; t++) {
      const hOff = t * hiddenDim;
      const outOff = t * llmDim;
      for (let oc = 0; oc < llmDim; oc++) {
        let sum = weights.mlp2Bias ? weights.mlp2Bias[oc] : 0.0;
        const wOff = oc * hiddenDim;
        for (let ic = 0; ic < hiddenDim; ic++) {
          sum += h1[hOff + ic] * weights.mlp2Weight[wOff + ic];
        }
        projected[outOff + oc] = sum;
      }
    }

    return projected;
  }

  /**
   * 2-Layer GeLU MLP 프로젝터 WebGPU 하드웨어 가속 순전파
   */
  public static async projectGPU(
    visualTokens: Float32Array,
    numTokens: number,
    weights: VLMProjectorWeights,
    hiddenDim: number = 2048,
    llmDim: number = 2048
  ): Promise<Float32Array> {
    const dev = getDevice();
    if (!dev) {
      throw new AMEVAForgeValidationError('[VLMProjector:WebGPU] WebGPU device is not available. Refusing silent fallback to CPU.');
    }

    const inDim = 768;
    const hTokens = uploadFloat32Array(visualTokens, [numTokens, inDim]);
    const hW1 = uploadFloat32Array(weights.mlp1Weight, [hiddenDim, inDim]);
    const hW2 = uploadFloat32Array(weights.mlp2Weight, [llmDim, hiddenDim]);
    const handles = [hTokens, hW1, hW2];

    try {
      // Linear 1: [numTokens, inDim] @ [inDim, hiddenDim] -> [numTokens, hiddenDim]
      const hW1T = gpuCore.transpose(hW1);
      handles.push(hW1T);
      const hH1 = gpuCore.matmul(hTokens, hW1T);
      handles.push(hH1);

      const rawH1 = await read(hH1);
      for (let t = 0; t < numTokens; t++) {
        const off = t * hiddenDim;
        for (let oc = 0; oc < hiddenDim; oc++) {
          const b = weights.mlp1Bias ? weights.mlp1Bias[oc] : 0.0;
          const v = rawH1[off + oc] + b;
          const clamped = Math.max(-88.0, Math.min(88.0, 1.702 * v));
          rawH1[off + oc] = v * (1.0 / (1.0 + Math.exp(-clamped)));
        }
      }

      // Linear 2: [numTokens, hiddenDim] @ [hiddenDim, llmDim] -> [numTokens, llmDim]
      const hActH1 = uploadFloat32Array(rawH1, [numTokens, hiddenDim]);
      handles.push(hActH1);
      const hW2T = gpuCore.transpose(hW2);
      handles.push(hW2T);
      const hOut = gpuCore.matmul(hActH1, hW2T);
      handles.push(hOut);

      const out = await read(hOut);
      if (weights.mlp2Bias) {
        for (let t = 0; t < numTokens; t++) {
          const off = t * llmDim;
          for (let oc = 0; oc < llmDim; oc++) {
            out[off + oc] += weights.mlp2Bias[oc];
          }
        }
      }

      return out;
    } finally {
      for (const h of handles) {
        try { dispose(h); } catch {}
      }
    }
  }
}

export const VLMEngine = VLMProjector;
