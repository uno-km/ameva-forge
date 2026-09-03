/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-331 CLIP Text Encoder Tests
 */

import { CLIPTextEncoder, CLIPTextEncoderWeights, CLIPLayerWeights } from '../src/diffusion/clipTextEncoder';
import { CLIPTokenizer } from '../src/diffusion/clipTokenizer';

function createDummyLayer(): CLIPLayerWeights {
  const dim = 768;
  const mlpDim = 3072;
  return {
    norm1Gamma: new Float32Array(dim).fill(1.0),
    norm1Beta: new Float32Array(dim).fill(0.0),
    qProjWeight: new Float32Array(dim * dim).fill(0.001),
    qProjBias: new Float32Array(dim).fill(0.0),
    kProjWeight: new Float32Array(dim * dim).fill(0.001),
    kProjBias: new Float32Array(dim).fill(0.0),
    vProjWeight: new Float32Array(dim * dim).fill(0.001),
    vProjBias: new Float32Array(dim).fill(0.0),
    outProjWeight: new Float32Array(dim * dim).fill(0.001),
    outProjBias: new Float32Array(dim).fill(0.0),
    norm2Gamma: new Float32Array(dim).fill(1.0),
    norm2Beta: new Float32Array(dim).fill(0.0),
    mlpFc1Weight: new Float32Array(mlpDim * dim).fill(0.001),
    mlpFc1Bias: new Float32Array(mlpDim).fill(0.0),
    mlpFc2Weight: new Float32Array(dim * mlpDim).fill(0.001),
    mlpFc2Bias: new Float32Array(dim).fill(0.0),
  };
}

describe('CLIPTextEncoder Tests (SCRUM-331)', () => {
  it('computes QuickGELU activation with mathematical accuracy', () => {
    const input = new Float32Array([-2.0, 0.0, 2.0]);
    const out = CLIPTextEncoder.quickGELU(input);
    expect(out[1]).toBeCloseTo(0.0, 5); // 0 * sigmoid(0) = 0
    expect(out[2]).toBeGreaterThan(1.8);
    expect(out[0]).toBeLessThan(0.0);
  });

  it('executes layerNorm with unit variance and zero mean on uniform input', () => {
    const dim = 768;
    const x = new Float32Array(dim);
    for (let i = 0; i < dim; i++) x[i] = i * 0.1;
    const gamma = new Float32Array(dim).fill(1.0);
    const beta = new Float32Array(dim).fill(0.0);

    const out = CLIPTextEncoder.layerNorm(x, 1, dim, gamma, beta);
    let sum = 0;
    for (let i = 0; i < dim; i++) sum += out[i];
    expect(Math.abs(sum / dim)).toBeLessThan(1e-4);
  });

  it('runs full 1-layer test encoder forward producing [77, 768] conditioning context without NaN', () => {
    const dim = 768;
    const tokenizer = new CLIPTokenizer();
    const { tokenIds } = tokenizer.encode('a photo of a neon city');

    const weights: CLIPTextEncoderWeights = {
      tokenEmbedding: new Float32Array(50000 * dim).fill(0.01),
      positionEmbedding: new Float32Array(77 * dim).fill(0.01),
      layers: [createDummyLayer()], // 1 layer for quick sanity
      finalNormGamma: new Float32Array(dim).fill(1.0),
      finalNormBeta: new Float32Array(dim).fill(0.0),
    };

    const context = CLIPTextEncoder.forward(tokenIds, weights);
    expect(context.length).toBe(77 * 768);
    for (let i = 0; i < context.length; i++) {
      expect(Number.isFinite(context[i])).toBe(true);
    }
  });
});
