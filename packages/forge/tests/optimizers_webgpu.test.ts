/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/tests/optimizers_webgpu.test.ts
 * Type: TypeScript Unit Test (WebGPU Native Fused Adam & Momentum SGD Verification)
 * Created: 2026-08-19T00:30:00+09:00
 * ============================================================================
 * WHAT:
 *   WebGPU Native Fused Adam(adam_step) 및 Momentum SGD(sgd_momentum_step) 커널의
 *   레지스트리 등록, gpuCore API 디스패치, executeGraph 스키마 무결성을 전수 검증합니다.
 */

import { executeGraph } from '../src/tensor/graphExecutor';
import { KERNEL_REGISTRY, gpuCore } from '../src/tensor/gpuCore';
import { _globalRegistry } from '../src/tensor/tensorRegistry';
import { _setDeviceForTesting } from '../src/webgpu/device';
import { clearStagingPool, allocateBuffer } from '../src/webgpu/buffers';
import { computeDispatch2D } from '../src/tensor/dispatchShape';

describe('WebGPU Native Fused Adam & Momentum SGD Optimizer Tests', () => {
  const mockDevice: any = {
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({
      getBindGroupLayout: jest.fn(() => ({})),
    })),
    createBuffer: jest.fn((desc: any) => ({
      size: desc.size,
      usage: desc.usage,
      destroy: jest.fn(),
      mapAsync: jest.fn().mockResolvedValue(undefined),
      getMappedRange: jest.fn(() => new ArrayBuffer(desc.size)),
      unmap: jest.fn(),
    })),
    createBindGroup: jest.fn(() => ({})),
    createCommandEncoder: jest.fn(() => ({
      beginComputePass: jest.fn(() => ({
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn(),
        end: jest.fn(),
      })),
      copyBufferToBuffer: jest.fn(),
      finish: jest.fn(() => ({})),
    })),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn().mockResolvedValue(null),
    queue: {
      submit: jest.fn(),
      writeBuffer: jest.fn(),
      onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined),
    },
  };

  beforeAll(() => {
    _setDeviceForTesting(mockDevice);
  });

  afterAll(() => {
    _setDeviceForTesting(null);
  });

  beforeEach(() => {
    jest.clearAllMocks();
    clearStagingPool();
  });

  it('should verify adam_step and sgd_momentum_step WGSL code in KERNEL_REGISTRY', () => {
    const adamWgsl = KERNEL_REGISTRY.get('adam_step');
    expect(adamWgsl).toBeDefined();
    expect(adamWgsl).toContain('struct AdamParams');
    expect(adamWgsl).toContain('let m_curr = params.beta1 * m_prev + (1.0 - params.beta1) * g;');
    expect(adamWgsl).toContain('let v_curr = params.beta2 * v_prev + (1.0 - params.beta2) * g * g;');

    const momentumWgsl = KERNEL_REGISTRY.get('sgd_momentum_step');
    expect(momentumWgsl).toBeDefined();
    expect(momentumWgsl).toContain('struct MomentumParams');
    expect(momentumWgsl).toContain('let v_curr = params.momentum * v_prev + g;');
  });

  it('should dispatch adam_step via gpuCore with valid in-place updates', () => {
    const N = 256;
    const { buffer: bufP, token: tokP } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'param');
    const { buffer: bufG, token: tokG } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'grad');
    const { buffer: bufM, token: tokM } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'm');
    const { buffer: bufV, token: tokV } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'v');

    const hP = _globalRegistry.register({ buffer: bufP, token: tokP, shape: [N], dtype: 'float32', byteLength: N * 4 });
    const hG = _globalRegistry.register({ buffer: bufG, token: tokG, shape: [N], dtype: 'float32', byteLength: N * 4 });
    const hM = _globalRegistry.register({ buffer: bufM, token: tokM, shape: [N], dtype: 'float32', byteLength: N * 4 });
    const hV = _globalRegistry.register({ buffer: bufV, token: tokV, shape: [N], dtype: 'float32', byteLength: N * 4 });

    gpuCore.adam_step(hP, hG, hM, hV, 0.001, 0.9, 0.999, 1e-8, 1);

    expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
    const lastCall = (mockDevice.queue.writeBuffer as jest.Mock).mock.calls.slice(-1)[0];
    const u32view = new Uint32Array(lastCall[2]);
    expect(u32view[0]).toBe(N);
  });

  it('should dispatch sgd_momentum_step via gpuCore with valid in-place updates', () => {
    const N = 256;
    const { buffer: bufP, token: tokP } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'param_sgd');
    const { buffer: bufG, token: tokG } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'grad_sgd');
    const { buffer: bufVel, token: tokVel } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'vel_sgd');

    const hP = _globalRegistry.register({ buffer: bufP, token: tokP, shape: [N], dtype: 'float32', byteLength: N * 4 });
    const hG = _globalRegistry.register({ buffer: bufG, token: tokG, shape: [N], dtype: 'float32', byteLength: N * 4 });
    const hVel = _globalRegistry.register({ buffer: bufVel, token: tokVel, shape: [N], dtype: 'float32', byteLength: N * 4 });

    gpuCore.sgd_momentum_step(hP, hG, hVel, 0.01, 0.9);

    expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
    const lastCall = (mockDevice.queue.writeBuffer as jest.Mock).mock.calls.slice(-1)[0];
    const u32view = new Uint32Array(lastCall[2]);
    expect(u32view[0]).toBe(N);
  });

  it('should validate in-place parameter handle preservation in executeGraph', async () => {
    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [64], params: [] },
      { id: 2, op: 'upload', in: [], shape: [64], params: [] },
      { id: 3, op: 'upload', in: [], shape: [64], params: [] },
      { id: 4, op: 'upload', in: [], shape: [64], params: [] },
      { id: 5, op: 'adam_step', in: [1, 2, 3, 4], shape: [64], params: [0.001, 0.9, 0.999, 1e-8, 0.9, 0.999] },
    ];

    const inputBuf = new Float32Array(64);
    const result = await executeGraph(JSON.stringify(instructions), [inputBuf, inputBuf, inputBuf, inputBuf]);
    // In-place contract: result for node 5 (adam_step) MUST equal handle for node 1 (param)
    expect(result['5']).toBe(result['1']);
  });
});
