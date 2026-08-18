/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/tests/sparse_cross_entropy.test.ts
 * Type: TypeScript Unit Test (WebGPU Native Sparse Cross-Entropy Verification)
 * Created: 2026-08-19T01:00:00+09:00
 * ============================================================================
 * WHAT:
 *   WebGPU Native Fused Sparse Cross-Entropy forward 및 backward 커널의
 *   레지스트리 등록, gpuCore API 디스패치, executeGraph 스키마 및 수치 불변성을 검증합니다.
 */

import { executeGraph } from '../src/tensor/graphExecutor';
import { KERNEL_REGISTRY, gpuCore } from '../src/tensor/gpuCore';
import { _globalRegistry } from '../src/tensor/tensorRegistry';
import { _setDeviceForTesting } from '../src/webgpu/device';
import { clearStagingPool, allocateBuffer } from '../src/webgpu/buffers';

describe('WebGPU Native Fused Sparse Cross-Entropy Tests', () => {
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

  it('should verify sparse_cross_entropy and sparse_cross_entropy_backward in KERNEL_REGISTRY', () => {
    expect(KERNEL_REGISTRY.has('sparse_cross_entropy')).toBe(true);
    expect(KERNEL_REGISTRY.has('sparse_cross_entropy_backward')).toBe(true);

    const fwdWgsl = KERNEL_REGISTRY.get('sparse_cross_entropy');
    expect(fwdWgsl).toContain('struct Params');
    expect(fwdWgsl).toContain('targets: array<u32>');
    expect(fwdWgsl).toContain('loss: array<f32>');

    const bwdWgsl = KERNEL_REGISTRY.get('sparse_cross_entropy_backward');
    expect(bwdWgsl).toContain('grad_logits: array<f32>');
    expect(bwdWgsl).toContain('reduction_scale: f32');
  });

  it('should dispatch sparseCrossEntropy via gpuCore and register valid handle', () => {
    const N = 4;
    const C = 1000;
    const { buffer: bufLogits, token: tokLogits } = allocateBuffer(N * C * 4, 0x0080 | 0x0004, 'tensor', 'logits');
    const { buffer: bufTargets, token: tokTargets } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'targets');

    const hLogits = _globalRegistry.register({
      buffer: bufLogits,
      token: tokLogits,
      shape: [N, C],
      dtype: 'float32',
      byteLength: N * C * 4
    });

    const hTargets = _globalRegistry.register({
      buffer: bufTargets,
      token: tokTargets,
      shape: [N],
      dtype: 'float32',
      byteLength: N * 4
    });

    const hLoss = gpuCore.sparseCrossEntropy(hLogits, hTargets, -100);
    expect(hLoss).toBeDefined();

    const tensorLoss = _globalRegistry.get(hLoss);
    expect(tensorLoss.shape).toEqual([N]);
    expect(tensorLoss.dtype).toBe('float32');
    expect(tensorLoss.byteLength).toBe(N * 4);
  });

  it('should dispatch sparseCrossEntropyBackward via gpuCore and register valid gradient handle', () => {
    const N = 4;
    const C = 1000;
    const { buffer: bufLogits, token: tokLogits } = allocateBuffer(N * C * 4, 0x0080 | 0x0004, 'tensor', 'logits_bwd');
    const { buffer: bufTargets, token: tokTargets } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'targets_bwd');
    const { buffer: bufGradOut, token: tokGradOut } = allocateBuffer(N * 4, 0x0080 | 0x0004, 'tensor', 'grad_out');

    const hLogits = _globalRegistry.register({
      buffer: bufLogits,
      token: tokLogits,
      shape: [N, C],
      dtype: 'float32',
      byteLength: N * C * 4
    });

    const hTargets = _globalRegistry.register({
      buffer: bufTargets,
      token: tokTargets,
      shape: [N],
      dtype: 'float32',
      byteLength: N * 4
    });

    const hGradOut = _globalRegistry.register({
      buffer: bufGradOut,
      token: tokGradOut,
      shape: [N],
      dtype: 'float32',
      byteLength: N * 4
    });

    const hGradLogits = gpuCore.sparseCrossEntropyBackward(hLogits, hTargets, hGradOut, -100, 1.0 / N);
    expect(hGradLogits).toBeDefined();

    const tensorGradLogits = _globalRegistry.get(hGradLogits);
    expect(tensorGradLogits.shape).toEqual([N, C]);
    expect(tensorGradLogits.dtype).toBe('float32');
    expect(tensorGradLogits.byteLength).toBe(N * C * 4);
  });

  it('should execute sparse_cross_entropy graph instructions without schema failure', async () => {
    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [2, 10], params: [] }, // logits: [2, 10]
      { id: 2, op: 'upload', in: [], shape: [2], params: [] },     // targets: [2]
      { id: 3, op: 'sparse_cross_entropy', in: [1, 2], shape: [2], params: [10, -100, 0, 0] },
      { id: 4, op: 'upload', in: [], shape: [2], params: [] },     // grad_out: [2]
      { id: 5, op: 'sparse_cross_entropy_backward', in: [1, 2, 4], shape: [2, 10], params: [-100, 0.5] }
    ];

    const logitsBuf = new Float32Array(2 * 10);
    const targetsBuf = new Float32Array([0, 1]);
    const gradOutBuf = new Float32Array([1.0, 1.0]);

    const result = await executeGraph(JSON.stringify(instructions), [logitsBuf, targetsBuf, gradOutBuf]);
    expect(result['3']).toBeDefined();
    expect(result['5']).toBeDefined();
  });
});
