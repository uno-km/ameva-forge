import { REDUCE_AXES_WGSL } from '../src/tensor/kernels/reduce_axes.wgsl';
import { getAllowedKernelNames } from '../src/webgpu/shaderGuard';

describe('SCRUM-266 ~ SCRUM-270: Native WebGPU Multi-Axis Fused Reduction Architectural Contract', () => {
  it('registers reduce_axes in shaderGuard whitelist', () => {
    const allowed = getAllowedKernelNames();
    expect(allowed.has('reduce_axes')).toBe(true);
  });

  it('validates reduce_axes.wgsl parameter struct and 1-pass loop structure', () => {
    expect(REDUCE_AXES_WGSL).toContain('struct Params');
    expect(REDUCE_AXES_WGSL).toContain('num_out_elements: u32');
    expect(REDUCE_AXES_WGSL).toContain('reduction_size: u32');
    expect(REDUCE_AXES_WGSL).toContain('axes_mask: array<u32, 8>');
    expect(REDUCE_AXES_WGSL).toContain('for (var r = 0u; r < params.reduction_size; r = r + 1u)');
    expect(REDUCE_AXES_WGSL).toContain('@compute @workgroup_size(64)');
  });

  it('executes reduce_axes graph instruction via graphExecutor', async () => {
    const { executeGraph } = await import('../src/tensor/graphExecutor');
    // Input 3D tensor: shape [2, 3, 4] -> 24 elements
    const inData = new Float32Array(24);
    for (let i = 0; i < 24; i++) inData[i] = 1.0;

    // Reduce axes (0, 1) -> output shape [4], reduction_size = 2*3 = 6
    // out_elements = 4
    // in_rank = 3
    // in_shape = [2, 3, 4, 0,0,0,0,0]
    // in_strides = [12, 4, 1, 0,0,0,0,0]
    // out_strides = [1, 0,0,0,0,0,0,0]
    // axes_mask = [1, 1, 0, 0,0,0,0,0]
    const params = [
      6, // reduction_size
      3, // in_rank
      2, 3, 4, 0, 0, 0, 0, 0, // in_shape
      12, 4, 1, 0, 0, 0, 0, 0, // in_strides
      1, 0, 0, 0, 0, 0, 0, 0, // out_strides
      1, 1, 0, 0, 0, 0, 0, 0  // axes_mask
    ];

    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [2, 3, 4], params: [] },
      { id: 2, op: 'reduce_axes', in: [1], shape: [4], params }
    ];

    const res = await executeGraph(JSON.stringify(instructions), [inData]);
    expect(res[2]).toBeDefined();
  });
});
