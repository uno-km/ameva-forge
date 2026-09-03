/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-331 CLIP-ViT/L14 Text Encoder WebGPU/CPU Forward Engine
 *
 * WHAT: 77개 토큰 시퀀스를 UNet Cross-Attention용 [77, 768] 부동소수점 컨디셔닝 텐서로 변환하는 트랜스포머 인코더입니다.
 * WHY: 가짜 감쇠 수식을 박멸하고, 실제 텍스트 프롬프트로부터 의미론적 잠재 컨텍스트 벡터를 생성하기 위해 존재합니다.
 * HOW: Token+Position Embedding -> 12계층 Transformer (LayerNorm -> MultiHead Causal Self-Attention -> QuickGELU MLP) -> Final LayerNorm.
 */

export interface CLIPLayerWeights {
  norm1Gamma: Float32Array; // [768]
  norm1Beta: Float32Array;  // [768]
  qProjWeight: Float32Array; // [768, 768]
  qProjBias: Float32Array;   // [768]
  kProjWeight: Float32Array; // [768, 768]
  kProjBias: Float32Array;   // [768]
  vProjWeight: Float32Array; // [768, 768]
  vProjBias: Float32Array;   // [768]
  outProjWeight: Float32Array; // [768, 768]
  outProjBias: Float32Array;   // [768]
  norm2Gamma: Float32Array; // [768]
  norm2Beta: Float32Array;  // [768]
  mlpFc1Weight: Float32Array; // [3072, 768]
  mlpFc1Bias: Float32Array;   // [3072]
  mlpFc2Weight: Float32Array; // [768, 3072]
  mlpFc2Bias: Float32Array;   // [768]
}

export interface CLIPTextEncoderWeights {
  tokenEmbedding: Float32Array;    // [vocabSize, 768]
  positionEmbedding: Float32Array; // [77, 768]
  layers: CLIPLayerWeights[];      // 12 layers
  finalNormGamma: Float32Array;    // [768]
  finalNormBeta: Float32Array;     // [768]
}

export class CLIPTextEncoder {
  public static readonly EMBED_DIM = 768;
  public static readonly SEQ_LEN = 77;
  public static readonly NUM_HEADS = 12;
  public static readonly HEAD_DIM = 64; // 768 / 12

  /**
   * LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
   */
  public static layerNorm(
    x: Float32Array,
    seqLen: number,
    dim: number,
    gamma: Float32Array,
    beta: Float32Array,
    eps: number = 1e-5
  ): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < seqLen; i++) {
      const offset = i * dim;
      let sum = 0.0;
      for (let d = 0; d < dim; d++) {
        sum += x[offset + d];
      }
      const mean = sum / dim;

      let sqDiff = 0.0;
      for (let d = 0; d < dim; d++) {
        const diff = x[offset + d] - mean;
        sqDiff += diff * diff;
      }
      const variance = sqDiff / dim;
      const invStd = 1.0 / Math.sqrt(variance + eps);

      for (let d = 0; d < dim; d++) {
        out[offset + d] = (x[offset + d] - mean) * invStd * gamma[d] + beta[d];
      }
    }
    return out;
  }

  /**
   * QuickGELU: x * sigmoid(1.702 * x)
   */
  public static quickGELU(x: Float32Array): Float32Array {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      const sig = 1.0 / (1.0 + Math.exp(-Math.max(-88.0, Math.min(88.0, 1.702 * v))));
      out[i] = v * sig;
    }
    return out;
  }

  /**
   * Multi-Head Causal Self-Attention (12 heads, 768 dim, Causal Mask)
   */
  public static forwardCausalAttention(
    x: Float32Array,
    seqLen: number,
    dim: number,
    numHeads: number,
    qW: Float32Array, qB: Float32Array,
    kW: Float32Array, kB: Float32Array,
    vW: Float32Array, vB: Float32Array,
    outW: Float32Array, outB: Float32Array
  ): Float32Array {
    const headDim = Math.floor(dim / numHeads);
    const scale = 1.0 / Math.sqrt(headDim);

    // 1. Linear projections: Q, K, V [seqLen, dim]
    const q = this.linear(x, seqLen, dim, dim, qW, qB);
    const k = this.linear(x, seqLen, dim, dim, kW, kB);
    const v = this.linear(x, seqLen, dim, dim, vW, vB);

    const out = new Float32Array(seqLen * dim);

    // 2. Multi-Head Dot-Product with Causal Mask
    for (let h = 0; h < numHeads; h++) {
      const headOffset = h * headDim;

      for (let i = 0; i < seqLen; i++) {
        const qOffset = i * dim + headOffset;

        // Compute row scores up to position i (causal mask: j > i is -Infinity)
        let maxScore = -Infinity;
        const scores = new Float32Array(seqLen);

        for (let j = 0; j <= i; j++) {
          const kOffset = j * dim + headOffset;
          let dot = 0.0;
          for (let d = 0; d < headDim; d++) {
            dot += q[qOffset + d] * k[kOffset + d];
          }
          const s = dot * scale;
          scores[j] = s;
          if (s > maxScore) maxScore = s;
        }

        // Softmax over 0..i
        let expSum = 0.0;
        for (let j = 0; j <= i; j++) {
          const e = Math.exp(scores[j] - maxScore);
          scores[j] = e;
          expSum += e;
        }
        const invSum = 1.0 / (expSum + 1e-9);
        for (let j = 0; j <= i; j++) {
          scores[j] *= invSum;
        }

        // Weighted sum of V
        const outOffset = i * dim + headOffset;
        for (let d = 0; d < headDim; d++) {
          let val = 0.0;
          for (let j = 0; j <= i; j++) {
            val += scores[j] * v[j * dim + headOffset + d];
          }
          out[outOffset + d] = val;
        }
      }
    }

    // 3. Final linear out projection
    return this.linear(out, seqLen, dim, dim, outW, outB);
  }

  /**
   * Dense Linear: y = x W^T + b
   */
  public static linear(
    x: Float32Array,
    seqLen: number,
    inDim: number,
    outDim: number,
    w: Float32Array,
    b?: Float32Array
  ): Float32Array {
    const out = new Float32Array(seqLen * outDim);
    for (let i = 0; i < seqLen; i++) {
      const xOffset = i * inDim;
      const outOffset = i * outDim;

      for (let oc = 0; oc < outDim; oc++) {
        let sum = b ? b[oc] : 0.0;
        const wOffset = oc * inDim;
        for (let ic = 0; ic < inDim; ic++) {
          sum += x[xOffset + ic] * w[wOffset + ic];
        }
        out[outOffset + oc] = sum;
      }
    }
    return out;
  }

  /**
   * CLIP 텍스트 인코더 전체 순전파:
   * [77] 토큰 ID -> Token/Position Embedding -> 12 Transformer Layers -> Final LayerNorm -> [77, 768] 부동소수점 텐서
   */
  public static forward(
    tokenIds: Int32Array,
    weights: CLIPTextEncoderWeights
  ): Float32Array {
    const dim = CLIPTextEncoder.EMBED_DIM;
    const seqLen = CLIPTextEncoder.SEQ_LEN;

    if (tokenIds.length !== seqLen) {
      throw new Error(`[CLIPTextEncoder] tokenIds length must be exactly ${seqLen}, received ${tokenIds.length}`);
    }

    // 1. Token & Position Embedding
    const hidden = new Float32Array(seqLen * dim);
    for (let i = 0; i < seqLen; i++) {
      const tokenId = tokenIds[i];
      const tokenOffset = tokenId * dim;
      const posOffset = i * dim;
      const hOffset = i * dim;

      for (let d = 0; d < dim; d++) {
        const tEmb = tokenOffset + d < weights.tokenEmbedding.length ? weights.tokenEmbedding[tokenOffset + d] : 0.0;
        const pEmb = weights.positionEmbedding[posOffset + d];
        hidden[hOffset + d] = tEmb + pEmb;
      }
    }

    let h = hidden;

    // 2. 12 Transformer Encoder Layers
    for (let layerIdx = 0; layerIdx < weights.layers.length; layerIdx++) {
      const layer = weights.layers[layerIdx];

      // Self-Attention Block
      const norm1 = this.layerNorm(h, seqLen, dim, layer.norm1Gamma, layer.norm1Beta);
      const attnOut = this.forwardCausalAttention(
        norm1, seqLen, dim, CLIPTextEncoder.NUM_HEADS,
        layer.qProjWeight, layer.qProjBias,
        layer.kProjWeight, layer.kProjBias,
        layer.vProjWeight, layer.vProjBias,
        layer.outProjWeight, layer.outProjBias
      );

      // Residual Add 1: h + attnOut
      for (let i = 0; i < h.length; i++) {
        h[i] += attnOut[i];
      }

      // MLP Block
      const norm2 = this.layerNorm(h, seqLen, dim, layer.norm2Gamma, layer.norm2Beta);
      const mlpFc1 = this.linear(norm2, seqLen, dim, 3072, layer.mlpFc1Weight, layer.mlpFc1Bias);
      const gelu = this.quickGELU(mlpFc1);
      const mlpFc2 = this.linear(gelu, seqLen, 3072, dim, layer.mlpFc2Weight, layer.mlpFc2Bias);

      // Residual Add 2: h + mlpFc2
      for (let i = 0; i < h.length; i++) {
        h[i] += mlpFc2[i];
      }
    }

    // 3. Final LayerNorm
    const finalContext = this.layerNorm(h, seqLen, dim, weights.finalNormGamma, weights.finalNormBeta);

    return finalContext;
  }
}
