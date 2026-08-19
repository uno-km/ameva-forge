import { SLICE_WGSL } from '../src/tensor/kernels/slice.wgsl';
import { SLICE_BACKWARD_WGSL } from '../src/tensor/kernels/slice_backward.wgsl';
import { getAllowedKernelNames } from '../src/webgpu/shaderGuard';
import { _setDeviceForTesting } from '../src/webgpu/device';
import { clearStagingPool } from '../src/webgpu/buffers';

describe('SCRUM-261 ~ SCRUM-265: Native WebGPU Slicing Architectural Contract', () => {
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
      copyBufferToBuffer: jest.fn(),
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

  it('registers slice and slice_backward in shaderGuard whitelist', () => {
    const allowed = getAllowedKernelNames();
    expect(allowed.has('slice')).toBe(true);
    expect(allowed.has('slice_backward')).toBe(true);
  });

  it('validates slice.wgsl structure and parameter uniforms', () => {
    expect(SLICE_WGSL).toContain('struct Params');
    expect(SLICE_WGSL).toContain('starts: array<u32, 8>');
    expect(SLICE_WGSL).toContain('steps: array<u32, 8>');
    expect(SLICE_WGSL).toContain('in_strides: array<u32, 8>');
    expect(SLICE_WGSL).toContain('out_strides: array<u32, 8>');
    expect(SLICE_WGSL).toContain('@compute @workgroup_size(64)');
  });

  it('validates slice_backward.wgsl gradient accumulation structure', () => {
    expect(SLICE_BACKWARD_WGSL).toContain('struct Params');
    expect(SLICE_BACKWARD_WGSL).toContain('grad_x[in_idx] = grad_output[idx];');
  });

  it('executes slice graph instruction via graphExecutor', async () => {
    const { executeGraph } = await import('../src/tensor/graphExecutor');
    // Input 2D tensor: shape [4, 4], data = 0..15
    const inData = new Float32Array(16);
    for (let i = 0; i < 16; i++) inData[i] = i;

    // Slice [1:3, 1:4] -> shape [2, 3]
    // rank = 2
    // starts = [1, 1, 0,0,0,0,0,0]
    // steps = [1, 1, 0,0,0,0,0,0]
    // in_strides = [4, 1, 0,0,0,0,0,0]
    // out_strides = [3, 1, 0,0,0,0,0,0]
    const params = [
      2, // rank
      1, 1, 0, 0, 0, 0, 0, 0, // starts
      1, 1, 0, 0, 0, 0, 0, 0, // steps
      4, 1, 0, 0, 0, 0, 0, 0, // in_strides
      3, 1, 0, 0, 0, 0, 0, 0  // out_strides
    ];

    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [4, 4], params: [] },
      { id: 2, op: 'slice', in: [1], shape: [2, 3], params }
    ];

    const res = await executeGraph(JSON.stringify(instructions), [inData]);
    expect(res[2]).toBeDefined();
  });
});
