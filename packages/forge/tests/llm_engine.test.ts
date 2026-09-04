/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: LLM Modality Unit Tests (RoPE, RMSNorm, SwiGLU, KV-Cache)
 */

import { LLMEngine, LLMWeights, KVCache, LLMError, LLMErrorCode } from '../src/llm/llmEngine';

describe('LLM Modality Tests (SCRUM-334)', () => {
  it('applies RMSNorm stabilizing variance to unit scale', () => {
    const dim = 64;
    const x = new Float32Array(dim);
    for (let i = 0; i < dim; i++) x[i] = i * 0.5;
    const gamma = new Float32Array(dim).fill(1.0);

    const out = LLMEngine.rmsNorm(x, gamma, dim);
    let sumSq = 0.0;
    for (let i = 0; i < dim; i++) sumSq += out[i] * out[i];
    expect(Math.sqrt(sumSq / dim)).toBeCloseTo(1.0, 3);
  });

  it('rotates position coordinates with RoPE embedding', () => {
    const dim = 64;
    const headDim = 64;
    const x = new Float32Array(dim).fill(1.0);

    const out0 = LLMEngine.applyRoPE(x, 0, dim, headDim);
    const out10 = LLMEngine.applyRoPE(x, 10, dim, headDim);

    expect(out0[0]).toBeCloseTo(1.0, 5); // At pos 0, cos(0)=1, sin(0)=0
    expect(out10[0]).not.toBeCloseTo(1.0, 2); // Rotated at pos 10
  });

  it('computes SwiGLU fused activation without NaN', () => {
    const dim = 64;
    const hiddenDim = 128;
    const x = new Float32Array(dim).fill(0.2);
    const wGate = new Float32Array(hiddenDim * dim).fill(0.01);
    const wUp = new Float32Array(hiddenDim * dim).fill(0.01);
    const wDown = new Float32Array(dim * hiddenDim).fill(0.01);

    const out = LLMEngine.swiglu(x, dim, hiddenDim, wGate, wUp, wDown);
    expect(out.length).toBe(dim);
    for (let i = 0; i < dim; i++) {
      expect(Number.isFinite(out[i])).toBe(true);
    }
  });

  it('runs single token forward pass with KV-cache update', () => {
    const dim = 512;
    const vocabSize = 1000;
    const weights: LLMWeights = {
      tokenEmbedding: new Float32Array(vocabSize * dim).fill(0.01),
      layers: [{
        inputNormGamma: new Float32Array(dim).fill(1.0),
        qWeight: new Float32Array(dim * dim).fill(0.001),
        kWeight: new Float32Array(dim * dim).fill(0.001),
        vWeight: new Float32Array(dim * dim).fill(0.001),
        outWeight: new Float32Array(dim * dim).fill(0.001),
        postNormGamma: new Float32Array(dim).fill(1.0),
        gateWeight: new Float32Array(1024 * dim).fill(0.001),
        upWeight: new Float32Array(1024 * dim).fill(0.001),
        downWeight: new Float32Array(dim * 1024).fill(0.001),
      }],
      finalNormGamma: new Float32Array(dim).fill(1.0),
      lmHeadWeight: new Float32Array(vocabSize * dim).fill(0.001),
    };

    const kvCaches: KVCache[] = [{
      k: new Float32Array(512 * dim),
      v: new Float32Array(512 * dim),
      length: 0,
    }];

    const logits = LLMEngine.forwardToken(42, 0, weights, kvCaches, dim, vocabSize);
    expect(logits.length).toBe(vocabSize);
    expect(kvCaches[0].length).toBe(1);
    for (let i = 0; i < vocabSize; i++) {
      expect(Number.isFinite(logits[i])).toBe(true);
    }
  });
});
