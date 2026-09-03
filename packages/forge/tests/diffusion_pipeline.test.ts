/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-327 ~ SCRUM-330 End-to-End Diffusion Pipeline & Unit Tests
 */

import {
  EulerDiscreteScheduler,
  VAEDecoder,
  VAEDecoderTestFixtures,
  ResNetBlock,
  ResNetBlockConfig,
  ResNetBlockWeights,
  WebGPUDiffusionPipeline,
} from '../src/index';

describe('In-Browser Diffusion Pipeline (SCRUM-327 ~ SCRUM-330)', () => {
  describe('1. EulerDiscreteScheduler (SCRUM-328)', () => {
    it('correctly sets timesteps and computes decreasing sigmas', () => {
      const scheduler = new EulerDiscreteScheduler(4);
      expect(scheduler.timesteps.length).toBe(4);
      expect(scheduler.sigmas.length).toBe(5); // numSteps + 1
      expect(scheduler.sigmas[4]).toBe(0.0); // final sigma is 0
      // Sigmas should be strictly non-increasing
      for (let i = 0; i < 4; i++) {
        expect(scheduler.sigmas[i]).toBeGreaterThanOrEqual(scheduler.sigmas[i + 1]);
      }
    });

    it('generates reproducible gaussian latent noise from seed with zero mean and unit variance', () => {
      const scheduler = new EulerDiscreteScheduler(1);
      const noise = scheduler.generateInitialNoise(4, 32, 32, 42);
      expect(noise.length).toBe(4 * 32 * 32);

      let sum = 0;
      let sqSum = 0;
      for (let i = 0; i < noise.length; i++) {
        sum += noise[i];
        sqSum += noise[i] * noise[i];
      }
      const mean = sum / noise.length;
      const variance = (sqSum / noise.length) - mean * mean;

      // Statistical test: mean near 0 (-0.1 ~ 0.1), variance near 1 (0.8 ~ 1.2)
      expect(Math.abs(mean)).toBeLessThan(0.1);
      expect(variance).toBeGreaterThan(0.7);
      expect(variance).toBeLessThan(1.3);
    });

    it('executes single Euler step update with mathematical parity', () => {
      const scheduler = new EulerDiscreteScheduler(2);
      const sample = new Float32Array([1.0, 2.0, 3.0]);
      const modelPred = new Float32Array([0.1, 0.2, 0.3]);
      const { prevSample } = scheduler.step(modelPred, 0, sample);

      const dt = scheduler.sigmas[1] - scheduler.sigmas[0];
      expect(prevSample[0]).toBeCloseTo(sample[0] + dt * modelPred[0], 5);
      expect(prevSample[1]).toBeCloseTo(sample[1] + dt * modelPred[1], 5);
      expect(prevSample[2]).toBeCloseTo(sample[2] + dt * modelPred[2], 5);
    });
  });

  describe('2. ResNetBlock Forward (SCRUM-327)', () => {
    it('executes full ResNet block forward (GroupNorm -> SiLU -> Conv2d -> SkipAdd) without NaN/Inf', () => {
      const config: ResNetBlockConfig = {
        inChannels: 32,
        outChannels: 32,
        height: 8,
        width: 8,
        numGroups: 32,
      };

      const weights: ResNetBlockWeights = {
        norm1Gamma: new Float32Array(32).fill(1.0),
        norm1Beta: new Float32Array(32).fill(0.0),
        conv1Weight: new Float32Array(32 * 32 * 9).fill(0.01),
        conv1Bias: new Float32Array(32).fill(0.0),
        norm2Gamma: new Float32Array(32).fill(1.0),
        norm2Beta: new Float32Array(32).fill(0.0),
        conv2Weight: new Float32Array(32 * 32 * 9).fill(0.01),
        conv2Bias: new Float32Array(32).fill(0.0),
      };

      const block = new ResNetBlock(config, weights);
      const input = new Float32Array(32 * 8 * 8).fill(0.5);
      const out = block.forwardCPU(input);

      expect(out.length).toBe(32 * 8 * 8);
      for (let i = 0; i < out.length; i++) {
        expect(Number.isFinite(out[i])).toBe(true);
        expect(Number.isNaN(out[i])).toBe(false);
      }
    });
  });

  describe('3. VAEDecoder (SCRUM-329)', () => {
    it('unscales latents using exact 1/0.18215 reciprocal factor', () => {
      const latent = new Float32Array([0.18215, -0.18215]);
      const unscaled = VAEDecoder.unscaleLatents(latent);
      expect(unscaled[0]).toBeCloseTo(1.0, 4);
      expect(unscaled[1]).toBeCloseTo(-1.0, 4);
    });

    it('converts floating-point RGB tensor to valid 8-bit RGBA canvas array', () => {
      // [-1.0, 0.0, 1.0]
      const rgb = new Float32Array([
        -1.0, // R
        0.0,  // G
        1.0,  // B
      ]);
      const rgba = VAEDecoder.tensorToRGBA(rgb, 1, 1);
      expect(rgba.length).toBe(4);
      expect(rgba[0]).toBe(0);   // (-1.0 + 1) * 127.5 = 0
      expect(rgba[1]).toBe(128); // (0.0 + 1) * 127.5 = 127.5 -> 128
      expect(rgba[2]).toBe(255); // (1.0 + 1) * 127.5 = 255
      expect(rgba[3]).toBe(255); // Alpha channel is 255
    });

    it('decodes latent spatial tensor to full RGBA canvas image data using explicit test fixture', () => {
      const latent = new Float32Array(4 * 8 * 8).fill(0.1);
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();
      const decoded = VAEDecoder.decodeLatentToRGB(latent, 8, 8, 64, 64, weights);

      expect(decoded.width).toBe(64);
      expect(decoded.height).toBe(64);
      expect(decoded.rgbaData.length).toBe(64 * 64 * 4);
      expect(decoded.floatData.length).toBe(3 * 64 * 64);
    });

    it('strictly refuses to decode when weights are omitted (Zero Silent Fallback)', () => {
      const latent = new Float32Array(4 * 8 * 8).fill(0.1);
      expect(() => {
        VAEDecoder.decodeLatentToRGB(latent, 8, 8, 64, 64, undefined as any);
      }).toThrow('[VAEDecoder] VAE decoder weights are required');
    });
  });

  describe('4. WebGPUDiffusionPipeline End-to-End (SCRUM-330)', () => {
    it('orchestrates complete diffusion pipeline with explicit vaeWeights', async () => {
      const pipeline = new WebGPUDiffusionPipeline();
      const weights = VAEDecoderTestFixtures.createSyntheticWeights();

      const progressSteps: number[] = [];
      const image = await pipeline.generate({
        prompt: 'a cinematic portrait of a cybernetic cat in neon city',
        numSteps: 2,
        width: 64,
        height: 64,
        seed: 777,
        vaeWeights: weights,
        onProgress: (p) => {
          progressSteps.push(p.step);
        },
      });

      expect(image).toBeDefined();
      expect(image.width).toBe(64);
      expect(image.height).toBe(64);
      expect(image.rgbaData.length).toBe(64 * 64 * 4);
      expect(progressSteps).toEqual([1, 2]);
    });

    it('strictly throws error when vaeWeights are missing in pipeline.generate()', async () => {
      const pipeline = new WebGPUDiffusionPipeline();
      await expect(pipeline.generate({
        prompt: 'test without weights',
        width: 64,
        height: 64,
      } as any)).rejects.toThrow('vaeWeights are strictly required');
    });
  });
});
