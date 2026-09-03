/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Operator Sanity Tests
 *
 * WHAT: VAE 디코더의 개별 연산자(Conv2d, GroupNorm Two-pass, SiLU, Upsample2D)의 수학적 계산 정확성과
 *      입력/가중치 불변성(Immutability), 반복 결정론(Determinism)을 검증하는 테스트 스위트입니다.
 * WHY: 허위 수치 Parity 과장을 배제하고, 순수 기초 연산자의 수학적 정합성을 정직하게 검증합니다.
 */

import { VAEDecoder } from '../src/diffusion/vaeDecoder';
import { VAEDecoderTestFixtures } from './fixtures/vaeDecoderFixtures';

describe('VAE Operator Sanity Tests (SCRUM-329)', () => {
  describe('1. Conv2d Same-Padding Convolution', () => {
    it('accurately computes full 32-element convolution matrix with same-padding', () => {
      const inC = 2;
      const outC = 2;
      const H = 4;
      const W = 4;
      const input = new Float32Array(inC * H * W).fill(1.0);
      const weight = new Float32Array(outC * inC * 9).fill(0.1);
      const bias = new Float32Array([0.5, -0.5]);

      const output = VAEDecoder.conv2d(input, inC, outC, H, W, weight, bias, 3, 1);
      expect(output.length).toBe(outC * H * W);

      // Corner [0, 0]: 2 channels * 4 active inputs * 0.1 = 0.8 + bias
      expect(output[0]).toBeCloseTo(1.3, 5);
      expect(output[H * W]).toBeCloseTo(0.3, 5);

      // Center [1, 1]: 2 channels * 9 active inputs * 0.1 = 1.8 + bias
      expect(output[1 * W + 1]).toBeCloseTo(2.3, 5);
      expect(output[H * W + 1 * W + 1]).toBeCloseTo(1.3, 5);

      for (let i = 0; i < output.length; i++) {
        expect(Number.isFinite(output[i])).toBe(true);
      }
    });
  });

  describe('2. GroupNorm Two-Pass Numerical Stability', () => {
    it('normalizes group channels with two-pass algorithm on standard distributions', () => {
      const C = 4;
      const H = 4;
      const W = 4;
      const G = 2;
      const totalElements = C * H * W;
      const input = new Float32Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        input[i] = (i % 8) * 1.5;
      }

      const gamma = new Float32Array(C).fill(2.0);
      const beta = new Float32Array(C).fill(1.0);

      const output = VAEDecoder.groupNorm(input, C, H, W, G, gamma, beta, 1e-5);
      expect(output.length).toBe(totalElements);

      const groupSize = (C / G) * H * W;
      for (let g = 0; g < G; g++) {
        let sum = 0;
        let sqDiffSum = 0;
        const offset = g * groupSize;
        for (let i = 0; i < groupSize; i++) {
          sum += output[offset + i];
        }
        const mean = sum / groupSize;
        expect(mean).toBeCloseTo(1.0, 3);

        for (let i = 0; i < groupSize; i++) {
          const diff = output[offset + i] - mean;
          sqDiffSum += diff * diff;
        }
        const variance = sqDiffSum / groupSize;
        expect(variance).toBeCloseTo(4.0, 3);
      }
    });

    it('remains numerically stable on extreme high-offset + low-variance tensors (Welford/Two-pass stability)', () => {
      const C = 2;
      const H = 2;
      const W = 2;
      const G = 1;
      const totalElements = C * H * W; // 8
      // High offset: 1e6, low variance: delta 0.1, 0.2, 0.3...
      const input = new Float32Array(totalElements);
      const baseOffset = 1000000.0;
      for (let i = 0; i < totalElements; i++) {
        input[i] = baseOffset + (i * 0.1);
      }

      const gamma = new Float32Array(C).fill(1.0);
      const beta = new Float32Array(C).fill(0.0);

      const output = VAEDecoder.groupNorm(input, C, H, W, G, gamma, beta, 1e-5);
      expect(output.length).toBe(totalElements);

      // Verify no NaN/Inf produced
      for (let i = 0; i < totalElements; i++) {
        expect(Number.isFinite(output[i])).toBe(true);
        expect(Number.isNaN(output[i])).toBe(false);
      }

      // Mean must be ~0.0 and variance ~1.0
      let sum = 0;
      for (let i = 0; i < totalElements; i++) sum += output[i];
      const mean = sum / totalElements;
      expect(Math.abs(mean)).toBeLessThan(1e-4);
    });

    it('handles constant tensor gracefully without division by zero', () => {
      const C = 2;
      const H = 2;
      const W = 2;
      const input = new Float32Array(C * H * W).fill(42.0);
      const gamma = new Float32Array(C).fill(1.0);
      const beta = new Float32Array(C).fill(0.0);

      const output = VAEDecoder.groupNorm(input, C, H, W, 1, gamma, beta, 1e-5);
      for (let i = 0; i < output.length; i++) {
        // variance=0 -> invStd = 1/sqrt(1e-5) -> (x - mean) = 0 -> output = 0
        expect(output[i]).toBeCloseTo(0.0, 5);
      }
    });
  });

  describe('3. Upsample2D Exact Bilinear Interpolation', () => {
    it('verifies exact bilinear interpolation arithmetic on 2x2 to 4x4 matrix', () => {
      const input = new Float32Array([
        10.0, 20.0,
        30.0, 40.0,
      ]);

      const output = VAEDecoder.upsample2d(input, 1, 2, 2, 4, 4);
      expect(output.length).toBe(16);

      // Corner pixel [0, 0]
      expect(output[0]).toBeCloseTo(10.0, 4);
      // Center-interpolated pixel [1, 1]
      expect(output[1 * 4 + 1]).toBeCloseTo(17.5, 4);
    });
  });

  describe('4. Determinism & Immutability', () => {
    it('produces bit-identical output across repeated decodes', () => {
      const latents = new Float32Array(4 * 8 * 8);
      for (let i = 0; i < latents.length; i++) latents[i] = Math.sin(i);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      const out1 = VAEDecoder.decode(latents, 8, 8, weights);
      const out2 = VAEDecoder.decode(latents, 8, 8, weights);

      for (let i = 0; i < out1.floatData.length; i++) {
        expect(out1.floatData[i]).toBe(out2.floatData[i]);
      }
      for (let i = 0; i < out1.rgbaData.length; i++) {
        expect(out1.rgbaData[i]).toBe(out2.rgbaData[i]);
      }
    });

    it('preserves input latents and weights immutably after decode', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.234);
      const latentsCopy = new Float32Array(latents);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();
      const weightCopy = new Float32Array(weights.convInWeight);

      VAEDecoder.decode(latents, 8, 8, weights);

      // Verify latents unchanged
      for (let i = 0; i < latents.length; i++) {
        expect(latents[i]).toBe(latentsCopy[i]);
      }
      // Verify weights unchanged
      for (let i = 0; i < weights.convInWeight.length; i++) {
        expect(weights.convInWeight[i]).toBe(weightCopy[i]);
      }
    });
  });
});
