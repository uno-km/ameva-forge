/**
 * Native WebGPU Slice Compute Kernel
 * Computes arbitrary multi-dimensional slice views directly on GPU VRAM.
 */
export const SLICE_WGSL = `
struct Params {
  num_elements: u32,
  rank: u32,
  workgroups_x: u32,
  pad: u32,
  starts: array<u32, 8>,
  steps: array<u32, 8>,
  in_strides: array<u32, 8>,
  out_strides: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;
  for (var i = 0u; i < params.rank; i = i + 1u) {
    let out_stride = max(params.out_strides[i], 1u);
    let coord = temp / out_stride;
    temp = temp % out_stride;

    let in_coord = params.starts[i] + coord * params.steps[i];
    in_idx = in_idx + in_coord * params.in_strides[i];
  }
  output[idx] = input[in_idx];
}
`;
