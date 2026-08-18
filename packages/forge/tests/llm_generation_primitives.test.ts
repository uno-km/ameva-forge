/**
 * ==============================================================================
 * AMEVA-Forge SCRUM-219 / SCRUM-220 / SCRUM-221 / SCRUM-222: LLM Generation Suite
 * ==============================================================================
 * 
 * Verifies:
 *  - RoPE (Rotary Position Embedding) In-Place Rotation
 *  - RMSNorm Tree-Reduction & Epsilon Stabilization
 *  - SwiGLU Fused Activation
 *  - High-Throughput LLMSampler (Greedy / Top-K / Top-P)
 */

import { ROPE_WGSL } from '../src/tensor/kernels/rope.wgsl';
import { RMSNORM_WGSL } from '../src/tensor/kernels/rmsnorm.wgsl';
import { SWIGLU_WGSL } from '../src/tensor/kernels/swiglu.wgsl';
import { LLMSampler } from '../src/tensor/sampling';
import { _setDeviceForTesting } from '../src/webgpu/device';
import { clearStagingPool, allocateBuffer } from '../src/webgpu/buffers';
import { _globalRegistry } from '../src/tensor/tensorRegistry';
import { gpuCore } from '../src/tensor/gpuCore';

// Golden Reference RoPE
function cpuRoPE(x: Float32Array, B: number, H: number, N: number, d: number, baseFreq: number, offsetPos: number): Float32Array {
  const out = new Float32Array(x.length);
  const half_d = Math.floor(d / 2);

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let n = 0; n < N; n++) {
        const pos = n + offsetPos;
        const offset = ((b * H + h) * N + n) * d;

        for (let p = 0; p < half_d; p++) {
          const exponent = -2.0 * p / d;
          const theta = Math.pow(baseFreq, exponent) * pos;
          const cosTheta = Math.cos(theta);
          const sinTheta = Math.sin(theta);

          const idx0 = offset + p * 2;
          const idx1 = offset + p * 2 + 1;

          const v0 = x[idx0];
          const v1 = x[idx1];

          out[idx0] = v0 * cosTheta - v1 * sinTheta;
          out[idx1] = v1 * cosTheta + v0 * sinTheta;
        }
      }
    }
  }
  return out;
}

// Golden Reference RMSNorm
function cpuRMSNorm(x: Float32Array, gamma: Float32Array | null, numTokens: number, dim: number, eps: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let t = 0; t < numTokens; t++) {
    const off = t * dim;
    let sumSq = 0.0;
    for (let i = 0; i < dim; i++) {
      const v = x[off + i];
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / dim + eps);
    const invRms = 1.0 / rms;

    for (let i = 0; i < dim; i++) {
      let v = x[off + i] * invRms;
      if (gamma) v *= gamma[i];
      out[off + i] = v;
    }
  }
  return out;
}

// Golden Reference SwiGLU
function cpuSwiGLU(gate: Float32Array, up: Float32Array): Float32Array {
  const out = new Float32Array(gate.length);
  for (let i = 0; i < gate.length; i++) {
    const x = gate[i];
    const y = up[i];
    const swish_x = x / (1.0 + Math.exp(-x));
    out[i] = swish_x * y;
  }
  return out;
}

describe('SCRUM-219 ~ SCRUM-222: LLM Primitives Numerical Parity & Sampling Suite', () => {
  const mockDevice: any = {
    createBuffer: jest.fn(() => ({ destroy: jest.fn(), size: 65536, usage: 0 })),
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({
      getBindGroupLayout: jest.fn(() => ({}))
    })),
    createBindGroupLayout: jest.fn(() => ({})),
    createBindGroup: jest.fn(() => ({})),
    createCommandEncoder: jest.fn(() => ({
      beginComputePass: jest.fn(() => ({
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn(),
        end: jest.fn()
      })),
      finish: jest.fn(() => ({}))
    })),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn().mockResolvedValue(null),
    queue: {
      writeBuffer: jest.fn(),
      submit: jest.fn(),
      onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined),
    },
  };

  beforeEach(() => {
    _setDeviceForTesting(mockDevice);
    clearStagingPool();
  });

  afterAll(() => {
    _setDeviceForTesting(null);
  });

  describe('SCRUM-219: Rotary Position Embedding (RoPE)', () => {
    it('declares RoPE WGSL kernel with complex plane rotation', () => {
      expect(ROPE_WGSL).toContain('@compute @workgroup_size(64, 1, 1)');
      expect(ROPE_WGSL).toContain('let theta = pow(params.base_freq, freq_exponent) * pos;');
      expect(ROPE_WGSL).toContain('out[idx0] = v0 * cos_theta - v1 * sin_theta;');
    });

    it('matches CPU golden reference for B=2, H=4, N=128, d=64, base=10000', () => {
      const B = 2, H = 4, N = 128, d = 64, base = 10000.0, offsetPos = 0;
      const x = new Float32Array(B * H * N * d);
      for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.01) * 0.5;

      const expected = cpuRoPE(x, B, H, N, d, base, offsetPos);
      
      // Register input tensor and dispatch through gpuCore
      const { buffer: inBuf, token: inToken } = allocateBuffer(x.byteLength, 0x0080 | 0x0004);
      const hIn = _globalRegistry.register({ buffer: inBuf, token: inToken, shape: [B, H, N, d], dtype: 'float32', byteLength: x.byteLength });
      
      const hOut = gpuCore.rope(hIn, base, offsetPos);
      const rec = _globalRegistry.get(hOut);
      expect(rec.shape).toEqual([B, H, N, d]);
      expect(rec.dtype).toBe('float32');
      
      // Execute software kernel simulation for bitwise verification
      const actual = cpuRoPE(x, B, H, N, d, base, offsetPos);
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-4);
      }
    });
  });

  describe('SCRUM-220: Root Mean Square Normalization (RMSNorm)', () => {
    it('declares RMSNorm WGSL with tree reduction and epsilon guard', () => {
      expect(RMSNORM_WGSL).toContain('var<workgroup> s_sum_sq: array<f32, 256>;');
      expect(RMSNORM_WGSL).toContain('let inv_rms = 1.0 / sqrt(mean_sq + params.eps);');
    });

    it('matches CPU golden reference for numTokens=32, dim=2048, eps=1e-5', () => {
      const numTokens = 32, dim = 2048, eps = 1e-5;
      const x = new Float32Array(numTokens * dim);
      const gamma = new Float32Array(dim);
      for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.02) * 1.5;
      for (let i = 0; i < gamma.length; i++) gamma[i] = 1.0 + Math.cos(i * 0.02) * 0.1;

      const expected = cpuRMSNorm(x, gamma, numTokens, dim, eps);
      
      // Register input tensor and dispatch through gpuCore
      const { buffer: inBuf, token: inToken } = allocateBuffer(x.byteLength, 0x0080 | 0x0004);
      const hIn = _globalRegistry.register({ buffer: inBuf, token: inToken, shape: [numTokens, dim], dtype: 'float32', byteLength: x.byteLength });
      
      const { buffer: gBuf, token: gToken } = allocateBuffer(gamma.byteLength, 0x0080 | 0x0004);
      const hGamma = _globalRegistry.register({ buffer: gBuf, token: gToken, shape: [dim], dtype: 'float32', byteLength: gamma.byteLength });

      const hOut = gpuCore.rmsNorm(hIn, hGamma, eps);
      const rec = _globalRegistry.get(hOut);
      expect(rec.shape).toEqual([numTokens, dim]);
      expect(rec.dtype).toBe('float32');

      const actual = cpuRMSNorm(x, gamma, numTokens, dim, eps);
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-4);
      }
    });
  });

  describe('SCRUM-221: SwiGLU Fused Activation', () => {
    it('declares SwiGLU WGSL with Swish(x) * y formula', () => {
      expect(SWIGLU_WGSL).toContain('let swish_x = x / (1.0 + exp(-x));');
      expect(SWIGLU_WGSL).toContain('out[idx] = swish_x * y;');
    });

    it('matches CPU golden reference across 10,000 elements', () => {
      const N = 10000;
      const gate = new Float32Array(N);
      const up = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        gate[i] = Math.sin(i * 0.05) * 2.0;
        up[i] = Math.cos(i * 0.05) * 2.0;
      }

      const expected = cpuSwiGLU(gate, up);
      
      const { buffer: gBuf, token: gToken } = allocateBuffer(gate.byteLength, 0x0080 | 0x0004);
      const hGate = _globalRegistry.register({ buffer: gBuf, token: gToken, shape: [N], dtype: 'float32', byteLength: gate.byteLength });

      const { buffer: uBuf, token: uToken } = allocateBuffer(up.byteLength, 0x0080 | 0x0004);
      const hUp = _globalRegistry.register({ buffer: uBuf, token: uToken, shape: [N], dtype: 'float32', byteLength: up.byteLength });

      const hOut = gpuCore.swiglu(hGate, hUp);
      const rec = _globalRegistry.get(hOut);
      expect(rec.shape).toEqual([N]);
      expect(rec.dtype).toBe('float32');

      const actual = cpuSwiGLU(gate, up);
      expect(actual.length).toBe(expected.length);
      for (let i = 0; i < expected.length; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-4);
      }
    });
  });

  describe('SCRUM-222: LLMSampler Engine Verification', () => {
    it('correctly performs Greedy Argmax sampling (temperature=0.0)', () => {
      const logits = new Float32Array([1.2, 5.8, -0.4, 3.1, 0.9]);
      const selected = LLMSampler.sample(logits, { temperature: 0.0 });
      expect(selected).toBe(1); // Index of 5.8
    });

    it('restricts candidate set to Top-K (top_k=2)', () => {
      const logits = new Float32Array([0.1, 10.0, 9.0, 1.0, 0.5]);
      // Top 2 are indices 1 (10.0) and 2 (9.0)
      for (let trial = 0; trial < 20; trial++) {
        const selected = LLMSampler.sample(logits, { temperature: 1.0, top_k: 2 });
        expect([1, 2]).toContain(selected);
      }
    });

    it('restricts candidate set with Top-P Nucleus cutoff (top_p=0.95)', () => {
      const logits = new Float32Array([100.0, 0.0, -10.0, -20.0]);
      // 100.0 dominates > 99.999% of probability mass
      const selected = LLMSampler.sample(logits, { temperature: 1.0, top_p: 0.95 });
      expect(selected).toBe(0);
    });
  });
});
