/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Strict Error Boundary & Fault Injection Tests
 *
 * WHAT: VAE 디코더의 모든 비정상 입력(가중치 누락, 초과/부족 upBlocks, NaN/Inf 가중치, 잘못된 차원, 리소스 초과)에 대해
 *      정확한 VAEDecoderErrorCode를 분출하는지 검증하는 결함 주입(Fault Injection) 테스트 스위트입니다.
 * WHY: 어떠한 침묵 폴백이나 가짜 데이터 은폐도 발생하지 않음을 증명합니다.
 */

import {
  VAEDecoder,
  VAEDecoderError,
  VAEDecoderErrorCode,
  VAE_DECODER_CAPABILITY,
  VAE_DECODER_ARCHITECTURE,
} from '../src/diffusion/vaeDecoder';
import { VAEDecoderTestFixtures } from './fixtures/vaeDecoderFixtures';

describe('VAE Strict Error Boundary & Fault Injection Tests (SCRUM-329)', () => {
  describe('1. Architecture & Capability Disclosure', () => {
    it('discloses honest prototype capabilities without claiming unverified AutoencoderKL parity', () => {
      expect(VAE_DECODER_CAPABILITY.autoencoder_kl_compatible).toBe(false);
      expect(VAE_DECODER_CAPABILITY.supports_real_checkpoint).toBe(false);
      expect(VAE_DECODER_CAPABILITY.numerical_parity_verified).toBe(false);
      expect(VAE_DECODER_CAPABILITY.architecture).toBe('fixed-3-stage-convolutional');
      expect(VAE_DECODER_ARCHITECTURE.upBlockCount).toBe(3);
    });
  });

  describe('2. Weights & Stage Count Boundaries (P0-1 & P0-2)', () => {
    it('throws VAE_WEIGHTS_REQUIRED when weights are null/undefined', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      expect(() => {
        VAEDecoder.decode(latents, 8, 8, undefined as any);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_WEIGHTS_REQUIRED,
      }));
    });

    it('rejects under-sized upBlocks (< 3 stages) with VAE_UPBLOCK_COUNT_MISMATCH', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();
      weights.upBlocks.pop(); // now 2 stages

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_UPBLOCK_COUNT_MISMATCH,
      }));
    });

    it('rejects extra upBlocks (> 3 stages) instead of silently ignoring them (P0-1 strict exactly 3)', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();
      weights.upBlocks.push(weights.upBlocks[0]); // now 4 stages

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_UPBLOCK_COUNT_MISMATCH,
      }));
    });
  });

  describe('3. Pre-Forward Finiteness Checks (P0-7 Fail-Fast on Weights)', () => {
    it('throws VAE_NON_FINITE_WEIGHT when convInWeight contains NaN before executing conv', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();
      weights.convInWeight[42] = NaN;

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_NON_FINITE_WEIGHT,
        message: expect.stringContaining('convInWeight contains non-finite value at index 42'),
      }));
    });

    it('throws VAE_NON_FINITE_WEIGHT when stage 2 normBeta contains Infinity', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();
      weights.upBlocks[2].normBeta[5] = Infinity;

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_NON_FINITE_WEIGHT,
        message: expect.stringContaining('upBlocks[2].normBeta contains non-finite value at index 5'),
      }));
    });

    it('throws VAE_NON_FINITE_INPUT when input latents contain NaN', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      latents[12] = NaN;
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_NON_FINITE_INPUT,
      }));
    });
  });

  describe('4. Scale Factor Validation (P0-5)', () => {
    it('rejects non-positive and non-finite scale factors', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      for (const invalid of [0, -0.18215, NaN, Infinity, -Infinity]) {
        expect(() => {
          VAEDecoder.unscaleLatents(latents, invalid);
        }).toThrow(expect.objectContaining({
          code: VAEDecoderErrorCode.VAE_SCALE_FACTOR_INVALID,
        }));
      }
    });
  });

  describe('5. Conv2d Contract Boundaries (P0-4)', () => {
    it('rejects even kernel sizes', () => {
      const x = new Float32Array(2 * 4 * 4).fill(1.0);
      const w = new Float32Array(2 * 2 * 4).fill(0.1);
      expect(() => {
        VAEDecoder.conv2d(x, 2, 2, 4, 4, w, undefined, 2, 1);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_CONV_CONTRACT_INVALID,
      }));
    });

    it('rejects padding mismatch with same-padding contract', () => {
      const x = new Float32Array(2 * 4 * 4).fill(1.0);
      const w = new Float32Array(2 * 2 * 9).fill(0.1);
      expect(() => {
        VAEDecoder.conv2d(x, 2, 2, 4, 4, w, undefined, 3, 0); // kernel 3 requires padding 1
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_CONV_CONTRACT_INVALID,
      }));
    });
  });

  describe('6. GroupNorm Error Boundaries (P1-2)', () => {
    it('rejects non-positive or non-finite eps', () => {
      const x = new Float32Array(4 * 4 * 4).fill(1.0);
      const gamma = new Float32Array(4).fill(1.0);
      const beta = new Float32Array(4).fill(0.0);

      for (const badEps of [0, -1e-5, NaN, Infinity]) {
        expect(() => {
          VAEDecoder.groupNorm(x, 4, 4, 4, 2, gamma, beta, badEps);
        }).toThrow(expect.objectContaining({
          code: VAEDecoderErrorCode.VAE_EPS_INVALID,
        }));
      }
    });

    it('rejects C indivisible by G', () => {
      const x = new Float32Array(5 * 4 * 4).fill(1.0);
      const gamma = new Float32Array(5).fill(1.0);
      const beta = new Float32Array(5).fill(0.0);

      expect(() => {
        VAEDecoder.groupNorm(x, 5, 4, 4, 2, gamma, beta);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_GROUP_DIVISIBILITY_ERROR,
      }));
    });
  });

  describe('7. Dimension & Resource Limits (P0-6)', () => {
    it('rejects non-safe integer dimensions', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      expect(() => {
        VAEDecoder.decode(latents, 8.5, 8, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_INVALID_DIMENSION,
      }));
    });

    it('throws VAE_RESOURCE_LIMIT_EXCEEDED when tensor exceeds configured limits', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      // Set extremely small limit
      expect(() => {
        VAEDecoder.decode(latents, 8, 8, weights, undefined, { maxTensorElements: 100 });
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_RESOURCE_LIMIT_EXCEEDED,
      }));
    });

    it('throws VAE_OUTPUT_SCALE_MISMATCH when decodeLatentToRGB requests mismatched size', () => {
      const latents = new Float32Array(4 * 8 * 8).fill(1.0);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      expect(() => {
        VAEDecoder.decodeLatentToRGB(latents, 8, 8, 1024, 768, weights);
      }).toThrow(expect.objectContaining({
        code: VAEDecoderErrorCode.VAE_OUTPUT_SCALE_MISMATCH,
      }));
    });
  });

  describe('8. Production Export Inspection (P0-2)', () => {
    it('ensures VAEDecoderTestFixtures is NOT exported by the production entrypoint', async () => {
      const prodModule = await import('../src/index');
      expect((prodModule as any).VAEDecoderTestFixtures).toBeUndefined();
    });
  });
});
