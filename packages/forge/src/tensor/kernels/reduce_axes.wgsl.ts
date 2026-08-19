/**
 * Native WebGPU Multi-Axis Fused Reduction Kernel
 * Reduces arbitrary multiple dimensions simultaneously in a single 1-Pass GPU dispatch.
 */
export const REDUCE_AXES_WGSL = `
struct Params {
  num_out_elements: u32,
  reduction_size: u32,
  in_rank: u32,
  workgroups_x: u32,
  in_shape: array<u32, 8>,
  in_strides: array<u32, 8>,
  out_strides: array<u32, 8>,
  axes_mask: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let out_idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (out_idx >= params.num_out_elements) { return; }

  // 1. Decompose out_idx into unreduced coordinates and compute base_in_idx
  var temp_out = out_idx;
  var base_in_idx = 0u;
  var out_dim_idx = 0u;

  for (var i = 0u; i < params.in_rank; i = i + 1u) {
    if (params.axes_mask[i] == 0u) {
      let out_stride = max(params.out_strides[out_dim_idx], 1u);
      let coord = temp_out / out_stride;
      temp_out = temp_out % out_stride;
      base_in_idx = base_in_idx + coord * params.in_strides[i];
      out_dim_idx = out_dim_idx + 1u;
    }
  }

  // 2. Iterate over the multi-axis reduction space in a single fused loop
  var sum_val = 0.0;
  for (var r = 0u; r < params.reduction_size; r = r + 1u) {
    var temp_r = r;
    var red_in_offset = 0u;
    for (var i = 0u; i < params.in_rank; i = i + 1u) {
      if (params.axes_mask[i] == 1u) {
        let dim_size = max(params.in_shape[i], 1u);
        let coord = temp_r % dim_size;
        temp_r = temp_r / dim_size;
        red_in_offset = red_in_offset + coord * params.in_strides[i];
      }
    }
    sum_val = sum_val + input[base_in_idx + red_in_offset];
  }

  output[out_idx] = sum_val;
}
`;
