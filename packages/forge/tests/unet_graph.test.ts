/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-332 UNet Execution Graph Tests
 */

import { UNetGraph, UNetWeights, UNetBlockWeights, SpatialCrossAttentionWeights } from '../src/diffusion/unetGraph';
import { ResNetBlockWeights } from '../src/diffusion/resnetBlock';

function createDummyCrossAttn(c: number): SpatialCrossAttentionWeights {
  return {
    normGamma: new Float32Array(c).fill(1.0),
    normBeta: new Float32Array(c).fill(0.0),
    qWeight: new Float32Array(c * c).fill(0.01),
    qBias: new Float32Array(c).fill(0.0),
    kWeight: new Float32Array(c * 768).fill(0.01),
    kBias: new Float32Array(c).fill(0.0),
    vWeight: new Float32Array(c * 768).fill(0.01),
    vBias: new Float32Array(c).fill(0.0),
    outWeight: new Float32Array(c * c).fill(0.01),
    outBias: new Float32Array(c).fill(0.0),
  };
}

function createDummyResNet(c: number): ResNetBlockWeights {
  return {
    norm1Gamma: new Float32Array(c).fill(1.0),
    norm1Beta: new Float32Array(c).fill(0.0),
    conv1Weight: new Float32Array(c * c * 3 * 3).fill(0.01),
    timeEmbProjWeight: new Float32Array(c * (320 * 4)).fill(0.01),
    norm2Gamma: new Float32Array(c).fill(1.0),
    norm2Beta: new Float32Array(c).fill(0.0),
    conv2Weight: new Float32Array(c * c * 3 * 3).fill(0.01),
  };
}

describe('UNetGraph Tests (SCRUM-332)', () => {
  it('computes sinusoidal time embedding across half-dimensions', () => {
    const emb = UNetGraph.computeSinusoidalTimeEmbedding(500, 320);
    expect(emb.length).toBe(320);
    expect(emb[0]).toBeCloseTo(Math.sin(500), 4);
    expect(emb[160]).toBeCloseTo(Math.cos(500), 4);
  });

  it('runs forward cross-attention without NaN or infinite values', () => {
    const C = 32;
    const H = 4;
    const W = 4;
    const x = new Float32Array(C * H * W).fill(0.5);
    const context = new Float32Array(77 * 768).fill(0.2);
    const weights = createDummyCrossAttn(C);

    const out = UNetGraph.forwardCrossAttention(x, C, H, W, context, 77, 768, weights);
    expect(out.length).toBe(C * H * W);
    for (let i = 0; i < out.length; i++) {
      expect(Number.isFinite(out[i])).toBe(true);
    }
  });

  it('executes full UNet forward pass predicting 4-channel noise', () => {
    const cBase = 32;
    const H = 4;
    const W = 4;
    const sample = new Float32Array(4 * H * W).fill(0.1);
    const context = new Float32Array(77 * 768).fill(0.05);

    const weights: UNetWeights = {
      convInWeight: new Float32Array(cBase * 4 * 3 * 3).fill(0.01),
      timeMlp1Weight: new Float32Array(320 * 4 * 320).fill(0.01),
      timeMlp1Bias: new Float32Array(320 * 4).fill(0.0),
      timeMlp2Weight: new Float32Array(320 * 4 * 320 * 4).fill(0.01),
      timeMlp2Bias: new Float32Array(320 * 4).fill(0.0),
      downBlocks: [{
        resnets: [createDummyResNet(cBase)],
        attentions: [createDummyCrossAttn(cBase)],
      }],
      midBlock: {
        resnet1: createDummyResNet(cBase),
        attention: createDummyCrossAttn(cBase),
        resnet2: createDummyResNet(cBase),
      },
      upBlocks: [{
        resnets: [createDummyResNet(cBase)],
        attentions: [createDummyCrossAttn(cBase)],
      }],
      normOutGamma: new Float32Array(cBase).fill(1.0),
      normOutBeta: new Float32Array(cBase).fill(0.0),
      convOutWeight: new Float32Array(4 * cBase * 3 * 3).fill(0.01),
    };

    const predNoise = UNetGraph.forward(sample, 999, context, weights, H, W, cBase);
    expect(predNoise.length).toBe(4 * H * W);
    for (let i = 0; i < predNoise.length; i++) {
      expect(Number.isFinite(predNoise[i])).toBe(true);
    }
  });
});
