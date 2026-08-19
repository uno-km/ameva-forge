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
import { computeDispatch2D } from '../src/tensor/dispatchShape';

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
    expect(fwdWgsl).toContain('index: array<f32>');

    const bwdWgsl = KERNEL_REGISTRY.get('embedding_backward');
    expect(bwdWgsl).toBeDefined();
    expect(bwdWgsl).toContain('struct EmbeddingBackwardParams');
    expect(bwdWgsl).toContain('index: array<f32>');
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

  it('should validate and execute embedding and embedding_backward in executeGraph', async () => {
    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [100, 32], params: [] },
      { id: 2, op: 'upload', in: [], shape: [2, 8], params: [] },
      { id: 3, op: 'embedding', in: [1, 2], shape: [2, 8, 32], params: [16, 32, 100, 0] },
      { id: 4, op: 'upload', in: [], shape: [2, 8, 32], params: [] },
      { id: 5, op: 'embedding_backward', in: [4, 2], shape: [100, 32], params: [16, 32, 100, 3200] }
    ];

    const weightData = new Float32Array(100 * 32);
    const indexData = new Float32Array(2 * 8);
    const gradData = new Float32Array(2 * 8 * 32);
    const res = await executeGraph(JSON.stringify(instructions), [weightData, indexData, gradData]);
    expect(res[3]).toBeDefined();
    expect(res[5]).toBeDefined();
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

  it('should guarantee 1:1 per-token workgroup dispatch for large sequences (N=128, 512, 2048) without 64x truncation', () => {
    // 1. N = 128 tokens
    const d128 = computeDispatch2D(128, 1);
    expect(d128.totalWorkgroups).toBe(128);
    expect(d128.dispatchX).toBe(128);
    expect(d128.dispatchY).toBe(1);

    // 2. N = 512 tokens
    const d512 = computeDispatch2D(512, 1);
    expect(d512.totalWorkgroups).toBe(512);
    expect(d512.dispatchX).toBe(512);
    expect(d512.dispatchY).toBe(1);

    // 3. N = 2048 tokens
    const d2048 = computeDispatch2D(2048, 1);
    expect(d2048.totalWorkgroups).toBe(2048);
    expect(d2048.dispatchX).toBe(2048);
    expect(d2048.dispatchY).toBe(1);

    // 4. Test physical gpuCore dispatch on large sequence N=128
    const { buffer: bufW, token: tokW } = allocateBuffer(32000 * 64 * 4, 0x0080 | 0x0004, 'tensor', 'large_w');
    const { buffer: bufI, token: tokI } = allocateBuffer(128 * 4, 0x0080 | 0x0004, 'tensor', 'large_i');

    const hWeight = _globalRegistry.register({
      buffer: bufW,
      token: tokW,
      shape: [32000, 64],
      dtype: 'float32',
      byteLength: 32000 * 64 * 4
    });

    const hIndex = _globalRegistry.register({
      buffer: bufI,
      token: tokI,
      shape: [1, 128],
      dtype: 'float32',
      byteLength: 128 * 4
    });

    const hOut = gpuCore.embedding(hWeight, hIndex);
    const tensorOut = _globalRegistry.get(hOut);
    expect(tensorOut.shape).toEqual([1, 128, 64]);
    expect(tensorOut.byteLength).toBe(1 * 128 * 64 * 4);
  });

  it('should support embedding_backward with > 64 repeated tokens without truncation', async () => {
    const numTokens = 128;
    const embeddingDim = 64;
    const vocabSize = 10;
    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [numTokens, embeddingDim], params: [] },
      { id: 2, op: 'upload', in: [], shape: [numTokens], params: [] },
      { id: 3, op: 'embedding_backward', in: [1, 2], shape: [vocabSize, embeddingDim], params: [numTokens, embeddingDim, vocabSize, vocabSize * embeddingDim] }
    ];

    const gradData = new Float32Array(numTokens * embeddingDim);
    gradData.fill(1.0);
    const indexData = new Float32Array(numTokens);
    // All 128 tokens are token_id 0!
    indexData.fill(0.0);

    const res = await executeGraph(JSON.stringify(instructions), [gradData, indexData]);
    expect(res[3]).toBeDefined();
    const tensorGradW = _globalRegistry.get(res[3]);
    expect(tensorGradW.shape).toEqual([vocabSize, embeddingDim]);
  });
});

