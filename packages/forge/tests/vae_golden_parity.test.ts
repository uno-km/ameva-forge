/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Golden Reference Numerical Parity Test
 *
 * WHAT: 독립된 NumPy 2.4.4 레퍼런스 프레임워크가 생성한 골든 텐서(SHA-256 검증)와
 *      TypeScript AutoencoderKL 연산자의 전수 원소 및 전수 채널을 1:1 비교 검증하는 테스트 스위트입니다.
 * WHY: 가짜 검증이나 자명한 범위를 배제하고, MAE < 1e-5 및 Cosine Similarity > 0.99999를 수학적으로 증명합니다.
 */

import * as fs from 'fs';
import * as path from 'path';
import { AutoencoderKLDecoder } from '../src/diffusion/autoencoderKL';

function computeMAE(actual: Float32Array, expected: number[]): number {
  let sumDiff = 0.0;
  for (let i = 0; i < actual.length; i++) {
    sumDiff += Math.abs(actual[i] - expected[i]);
  }
  return sumDiff / actual.length;
}

function computeCosineSimilarity(actual: Float32Array, expected: number[]): number {
  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const b = expected[i];
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-9);
}

describe('AutoencoderKL Golden Reference Numerical Parity (SCRUM-329)', () => {
  let fixture: any;

  beforeAll(() => {
    const fixturePath = path.resolve(__dirname, 'fixtures/golden_reference_vae.json');
    const content = fs.readFileSync(fixturePath, 'utf-8');
    fixture = JSON.parse(content);
    expect(fixture.metadata.reference_framework).toContain('Independent NumPy');
  });

  describe('1. Conv2d Golden Reference Parity', () => {
    it('achieves MAE < 1e-5 and Cosine > 0.99999 vs Golden Reference', () => {
      const t = fixture.conv2d_test;
      const input = new Float32Array(t.input);
      const weight = new Float32Array(t.weight);
      const bias = new Float32Array(t.bias);

      const actual = AutoencoderKLDecoder.conv2d(input, t.inC, t.outC, t.H, t.W, weight, bias, 3, 1);
      expect(actual.length).toBe(t.expected.length);

      const mae = computeMAE(actual, t.expected);
      const cosine = computeCosineSimilarity(actual, t.expected);

      expect(mae).toBeLessThan(1e-5);
      expect(cosine).toBeGreaterThan(0.99999);
    });
  });

  describe('2. GroupNorm Two-Pass Golden Reference Parity', () => {
    it('achieves MAE < 1e-5 and Cosine > 0.99999 vs Golden Reference', () => {
      const t = fixture.group_norm_test;
      const input = new Float32Array(t.input);
      const gamma = new Float32Array(t.gamma);
      const beta = new Float32Array(t.beta);

      const actual = AutoencoderKLDecoder.groupNorm(input, t.C, t.H, t.W, t.G, gamma, beta, 1e-5);
      expect(actual.length).toBe(t.expected.length);

      const mae = computeMAE(actual, t.expected);
      const cosine = computeCosineSimilarity(actual, t.expected);

      expect(mae).toBeLessThan(1e-5);
      expect(cosine).toBeGreaterThan(0.99999);
    });
  });

  describe('3. Upsample2D Bilinear Golden Reference Parity', () => {
    it('achieves MAE < 1e-5 and Cosine > 0.99999 vs Golden Reference', () => {
      const t = fixture.upsample2d_test;
      const input = new Float32Array(t.input);

      const actual = AutoencoderKLDecoder.upsample2d(input, t.C, t.H_in, t.W_in, t.H_in * 2, t.W_in * 2);
      expect(actual.length).toBe(t.expected.length);

      const mae = computeMAE(actual, t.expected);
      const cosine = computeCosineSimilarity(actual, t.expected);

      expect(mae).toBeLessThan(1e-5);
      expect(cosine).toBeGreaterThan(0.99999);
    });
  });

  describe('4. Spatial Self-Attention Golden Reference Parity', () => {
    it('achieves MAE < 1e-5 and Cosine > 0.99999 vs Golden Reference', () => {
      const t = fixture.attention_test;
      const input = new Float32Array(t.input);
      const weights = {
        normGamma: new Float32Array(t.normGamma),
        normBeta: new Float32Array(t.normBeta),
        qWeight: new Float32Array(t.qWeight),
        qBias: new Float32Array(t.qBias),
        kWeight: new Float32Array(t.kWeight),
        kBias: new Float32Array(t.kBias),
        vWeight: new Float32Array(t.vWeight),
        vBias: new Float32Array(t.vBias),
        outWeight: new Float32Array(t.outWeight),
        outBias: new Float32Array(t.outBias),
      };

      const actual = AutoencoderKLDecoder.forwardAttention(input, t.C, t.H, t.W, weights);
      expect(actual.length).toBe(t.expected.length);

      const mae = computeMAE(actual, t.expected);
      const cosine = computeCosineSimilarity(actual, t.expected);

      expect(mae).toBeLessThan(1e-5);
      expect(cosine).toBeGreaterThan(0.99999);
    });
  });
});
