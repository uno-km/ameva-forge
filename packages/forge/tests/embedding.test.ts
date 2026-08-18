/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/tests/embedding.test.ts
 * Type: TypeScript Unit Test (WebGPU Embedding Forward & Backward Verification)
 * Created: 2026-08-18T23:37:00+09:00
 * ============================================================================
 * WHAT:
 *   WebGPU Native 임베딩(embedding) 및 임베딩 역전파(embedding_backward) 커널의
 *   레지스트리 등록, gpuCore API 디스패치, executeGraph 스키마 무결성을 전수 검증합니다.
 */

import { executeGraph } from '../src/tensor/graphExecutor';
import { KERNEL_REGISTRY, gpuCore } from '../src/tensor/gpuCore';
import { _globalRegistry } from '../src/tensor/tensorRegistry';
import { _setDeviceForTesting } from '../src/webgpu/device';
import { clearStagingPool, allocateBuffer } from '../src/webgpu/buffers';

describe('WebGPU Native Embedding & Embedding Backward Kernel Tests', () => {
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

  it('should have embedding and embedding_backward registered in KERNEL_REGISTRY', () => {
    expect(KERNEL_REGISTRY.has('embedding')).toBe(true);
    expect(KERNEL_REGISTRY.has('embedding_backward')).toBe(true);

    const fwdWgsl = KERNEL_REGISTRY.get('embedding');
    expect(fwdWgsl).toBeDefined();
    expect(fwdWgsl).toContain('struct EmbeddingParams');
    expect(fwdWgsl).toContain('index: array<u32>');

    const bwdWgsl = KERNEL_REGISTRY.get('embedding_backward');
    expect(bwdWgsl).toBeDefined();
    expect(bwdWgsl).toContain('struct EmbeddingBackwardParams');
    expect(bwdWgsl).toContain('index: array<u32>');
    expect(bwdWgsl).toContain('grad_weight: array<f32>');
  });

  it('should dispatch embedding via gpuCore and register valid handle', () => {
    const { buffer: bufW, token: tokW } = allocateBuffer(1000 * 32 * 4, 0x0080 | 0x0004, 'tensor', 'test_w');
    const { buffer: bufI, token: tokI } = allocateBuffer(16 * 4, 0x0080 | 0x0004, 'tensor', 'test_i');

    const hWeight = _globalRegistry.register({
      buffer: bufW,
      token: tokW,
      shape: [1000, 32],
      dtype: 'float32',
      byteLength: 1000 * 32 * 4
    });

    const hIndex = _globalRegistry.register({
      buffer: bufI,
      token: tokI,
      shape: [2, 8],
      dtype: 'float32',
      byteLength: 16 * 4
    });

    const hOut = gpuCore.embedding(hWeight, hIndex);
    expect(hOut).toBeDefined();
    expect(typeof hOut).toBe('string');
    expect(hOut.length).toBeGreaterThan(0);

    const tensorOut = _globalRegistry.get(hOut);
    expect(tensorOut).toBeDefined();
    expect(tensorOut.shape).toEqual([2, 8, 32]);
    expect(tensorOut.dtype).toBe('float32');
    expect(tensorOut.byteLength).toBe(2 * 8 * 32 * 4);
  });

  it('should dispatch embedding_backward via gpuCore and register valid gradient handle', () => {
    const { buffer: bufG, token: tokG } = allocateBuffer(16 * 32 * 4, 0x0080 | 0x0004, 'tensor', 'test_g');
    const { buffer: bufI, token: tokI } = allocateBuffer(16 * 4, 0x0080 | 0x0004, 'tensor', 'test_i_bwd');

    const hGradOut = _globalRegistry.register({
      buffer: bufG,
      token: tokG,
      shape: [2, 8, 32],
      dtype: 'float32',
      byteLength: 16 * 32 * 4
    });

    const hIndex = _globalRegistry.register({
      buffer: bufI,
      token: tokI,
      shape: [2, 8],
      dtype: 'float32',
      byteLength: 16 * 4
    });

    const hGradWeight = gpuCore.embedding_backward(hGradOut, hIndex, 1000, 32);
    expect(hGradWeight).toBeDefined();
    expect(typeof hGradWeight).toBe('string');
    expect(hGradWeight.length).toBeGreaterThan(0);

    const tensorGradW = _globalRegistry.get(hGradWeight);
    expect(tensorGradW).toBeDefined();
    expect(tensorGradW.shape).toEqual([1000, 32]);
    expect(tensorGradW.dtype).toBe('float32');
    expect(tensorGradW.byteLength).toBe(1000 * 32 * 4);
  });

  it('should validate embedding and embedding_backward schemas in executeGraph', async () => {
    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [100, 32], params: [] },
      { id: 2, op: 'upload', in: [], shape: [2, 8], params: [] },
      { id: 3, op: 'embedding', in: [1, 2], shape: [2, 8, 32], params: [16, 32, 100, 0] },
      { id: 4, op: 'upload', in: [], shape: [2, 8, 32], params: [] },
      { id: 5, op: 'embedding_backward', in: [4, 2], shape: [100, 32], params: [16, 32, 100, 3200] }
    ];

    try {
      await executeGraph(JSON.stringify(instructions), []);
    } catch (err: any) {
      expect(err.message).not.toContain('expected min');
      expect(err.message).not.toContain('Unknown operation');
    }
  });

  it('should reject embedding_backward instruction with insufficient inputs (< 2)', async () => {
    const invalidInstructions = [
      { id: 1, op: 'upload', in: [], shape: [2, 8, 32], params: [] },
      { id: 2, op: 'embedding_backward', in: [1], shape: [100, 32], params: [16, 32, 100, 3200] }
    ];

    await expect(executeGraph(JSON.stringify(invalidInstructions), [])).rejects.toThrow(
      /Instruction\[1\] op="embedding_backward": expected exact 2 inputs, got 1/
    );
  });
});
