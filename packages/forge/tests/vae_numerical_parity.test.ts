/**
 * 파일 생성일: 2026-09-03
 * 수정일: 2026-09-03 (P0 긴급 시정: 타우톨로지 테스트 박멸, 전수 수치 일치도 및 결함 주입 검증 구현)
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Decoder Component Sanity & Strict Error Boundary Tests
 *
 * WHAT: VAE 디코더의 수학적 기본 연산자(Conv2d, GroupNorm, SiLU, Upsample2D)의 수치 정확성과,
 *      가중치 누락, NaN 입력, 잘못된 그룹 설정에 대한 엄격한 예외 분출(Fail-Fast)을 검증하는 테스트 스위트입니다.
 * WHY: 가짜 가중치 침묵 폴백이나 NaN 검은 픽셀 은폐를 원천 차단하고,
 *      오류 발생 시 즉시 시스템이 중단됨을 결함 주입(Fault Injection)으로 증명하기 위함입니다.
 */

import { VAEDecoder, VAEDecoderTestFixtures } from '../src/diffusion/vaeDecoder';

describe('AutoencoderKL VAE Component Sanity & Strict Error Boundary Tests (SCRUM-329)', () => {
  describe('1. Conv2d Full Output Matrix Sanity', () => {
    it('accurately computes full 32-element convolution output without silent truncation', () => {
      const inC = 2;
      const outC = 2;
      const H = 4;
      const W = 4;
      const input = new Float32Array(inC * H * W).fill(1.0);
      const weight = new Float32Array(outC * inC * 9).fill(0.1);
      const bias = new Float32Array([0.5, -0.5]);

      const output = VAEDecoder.conv2d(input, inC, outC, H, W, weight, bias, 3, 1);
      expect(output.length).toBe(outC * H * W);

      // Verify corner [0, 0]: 2 channels * 4 active inputs * 0.1 = 0.8 + bias
      // ch0 corner: 0.8 + 0.5 = 1.3
      expect(output[0]).toBeCloseTo(1.3, 5);
      // ch1 corner: 0.8 - 0.5 = 0.3
      expect(output[H * W]).toBeCloseTo(0.3, 5);

      // Verify center [1, 1]: 2 channels * 9 active inputs * 0.1 = 1.8 + bias
      // ch0 center: 1.8 + 0.5 = 2.3
      expect(output[1 * W + 1]).toBeCloseTo(2.3, 5);
      // ch1 center: 1.8 - 0.5 = 1.3
      expect(output[H * W + 1 * W + 1]).toBeCloseTo(1.3, 5);

      // Verify all 32 elements are finite
      for (let i = 0; i < output.length; i++) {
        expect(Number.isFinite(output[i])).toBe(true);
      }
    });

    it('throws error when weight length is mismatched (prevents undefined * number = NaN)', () => {
      const input = new Float32Array(2 * 4 * 4).fill(1.0);
      const corruptedWeight = new Float32Array(10); // expected 2 * 2 * 9 = 36
      expect(() => {
        VAEDecoder.conv2d(input, 2, 2, 4, 4, corruptedWeight, undefined, 3, 1);
      }).toThrow('[VAEDecoder] weight in conv2d length mismatch');
    });
  });

  describe('2. GroupNorm Strict Boundaries & Parity', () => {
    it('normalizes each group independently and verifies both groups', () => {
      const C = 4;
      const H = 4;
      const W = 4;
      const G = 2; // 2 groups
      const totalElements = C * H * W;
      const input = new Float32Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        input[i] = (i % 8) * 1.5;
      }

      const gamma = new Float32Array(C).fill(2.0); // 2x scale
      const beta = new Float32Array(C).fill(1.0);  // +1 shift

      const output = VAEDecoder.groupNorm(input, C, H, W, G, gamma, beta, 1e-5);
      expect(output.length).toBe(totalElements);

      const groupSize = (C / G) * H * W;
      for (let g = 0; g < G; g++) {
        let sum = 0;
        let sqSum = 0;
        const offset = g * groupSize;
        for (let i = 0; i < groupSize; i++) {
          sum += output[offset + i];
          sqSum += output[offset + i] * output[offset + i];
        }
        const mean = sum / groupSize;
        // Since scale is 2.0 and shift is 1.0, mean should be ~1.0 and variance should be ~4.0
        expect(mean).toBeCloseTo(1.0, 3);
        const variance = sqSum / groupSize - mean * mean;
        expect(variance).toBeCloseTo(4.0, 3);
      }
    });

    it('throws error when C is not divisible by G (prevents silent trailing channel deletion)', () => {
      const input = new Float32Array(5 * 4 * 4).fill(1.0); // C = 5
      const gamma = new Float32Array(5).fill(1.0);
      const beta = new Float32Array(5).fill(0.0);

      expect(() => {
        VAEDecoder.groupNorm(input, 5, 4, 4, 2, gamma, beta); // 5 % 2 !== 0
      }).toThrow('[VAEDecoder] GroupNorm requires C divisible by G: C=5, G=2');
    });

    it('throws error when affine gamma/beta lengths do not match C', () => {
      const input = new Float32Array(4 * 4 * 4).fill(1.0);
      const badGamma = new Float32Array(2);
      const beta = new Float32Array(4).fill(0.0);

      expect(() => {
        VAEDecoder.groupNorm(input, 4, 4, 4, 2, badGamma, beta);
      }).toThrow('[VAEDecoder] GroupNorm affine parameter mismatch');
    });
  });

  describe('3. Upsample2D Precise Bilinear Mathematics', () => {
    it('verifies exact mathematical bilinear interpolation formula on 2x2 to 4x4 expansion', () => {
      const C = 1;
      const H_in = 2;
      const W_in = 2;
      const input = new Float32Array([
        10.0, 20.0,
        30.0, 40.0,
      ]);

      const output = VAEDecoder.upsample2d(input, C, H_in, W_in, 4, 4);
      expect(output.length).toBe(16);

      // Verify corner pixel [0, 0]: real_h = (0 + 0.5)/2 - 0.5 = -0.25 -> clamped h0=0, h1=0, dh=0 -> exactly 10.0
      expect(output[0]).toBeCloseTo(10.0, 4);
      // Verify center-interpolated pixel [1, 1]:
      // real_h = 1.5/2 - 0.5 = 0.25 -> h0=0, h1=1, dh=0.25
      // real_w = 1.5/2 - 0.5 = 0.25 -> w0=0, w1=1, dw=0.25
      // top = 10 * 0.75 + 20 * 0.25 = 12.5
      // bottom = 30 * 0.75 + 40 * 0.25 = 32.5
      // val = 12.5 * 0.75 + 32.5 * 0.25 = 9.375 + 8.125 = 17.5
      expect(output[1 * 4 + 1]).toBeCloseTo(17.5, 4);
    });
  });

  describe('4. Strict Error Boundaries & Fault Injection (P0 Zero Silent Failure)', () => {
    it('refuses to decode when weights object is null/undefined (Zero Silent Fallback)', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      expect(() => {
        VAEDecoder.decode(latents, 8, 8, undefined as any);
      }).toThrow('[VAEDecoder] VAE decoder weights are required. Refusing to decode with synthetic weights.');
    });

    it('refuses to decode when an upsample stage weight is missing (Zero Stage Fallback)', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const brokenWeights = VAEDecoderTestFixtures.createSyntheticWeights();
      // Corrupt stage 1
      brokenWeights.upBlocks[1].upsampleConvWeight = undefined as any;

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, brokenWeights);
      }).toThrow('[VAEDecoder] Missing VAE upsample convolution weight at stage 1');
    });

    it('throws error immediately when input contains NaN (prevents silent black pixel masking)', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      latents[15] = NaN; // Inject corrupted NaN value
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights);
      }).toThrow('[VAEDecoder] latents input contains non-finite value at index 15');
    });

    it('throws error when decodeLatentToRGB requests dimensions that mismatch 8x scaling', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      expect(() => {
        VAEDecoder.decodeLatentToRGB(latents, 8, 8, 1024, 768, weights);
      }).toThrow('[VAEDecoder] decodeLatentToRGB scale mismatch: requested 1024x768, but latent 8x8 scales to 64x64');
    });

    it('successfully decodes valid latent tensor with explicitly passed test fixtures', () => {
      const latents = new Float32Array(4 * 8 * 8);
      for (let i = 0; i < latents.length; i++) {
        latents[i] = Math.sin(i * 0.1) * 0.5;
      }
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      const decoded = VAEDecoder.decode(latents, 8, 8, weights);
      expect(decoded.width).toBe(64);
      expect(decoded.height).toBe(64);
      expect(decoded.rgbaData.length).toBe(64 * 64 * 4);

      // Verify that every single float value is strictly finite and not NaN
      for (let i = 0; i < decoded.floatData.length; i++) {
        expect(Number.isFinite(decoded.floatData[i])).toBe(true);
        expect(Number.isNaN(decoded.floatData[i])).toBe(false);
      }
    });
  });
});
