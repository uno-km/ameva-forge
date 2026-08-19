/**
 * Native WebGPU Multi-Axis Fused Reduction Kernel
 * Reduces arbitrary multiple dimensions simultaneously in a single 1-Pass GPU dispatch.
 * Fully compliant with W3C WebGPU 16-byte uniform alignment rules using scalar fields.
 */
export const REDUCE_AXES_WGSL = `
struct Params {
  num_out_elements: u32,
  reduction_size: u32,
  in_rank: u32,
  workgroups_x: u32,
  in_shape0: u32, in_shape1: u32, in_shape2: u32, in_shape3: u32,
  in_shape4: u32, in_shape5: u32, in_shape6: u32, in_shape7: u32,
  in_stride0: u32, in_stride1: u32, in_stride2: u32, in_stride3: u32,
  in_stride4: u32, in_stride5: u32, in_stride6: u32, in_stride7: u32,
  out_stride0: u32, out_stride1: u32, out_stride2: u32, out_stride3: u32,
  out_stride4: u32, out_stride5: u32, out_stride6: u32, out_stride7: u32,
  axes_mask0: u32, axes_mask1: u32, axes_mask2: u32, axes_mask3: u32,
  axes_mask4: u32, axes_mask5: u32, axes_mask6: u32, axes_mask7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn get_in_shape(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.in_shape0; }
    case 1u: { return params.in_shape1; }
    case 2u: { return params.in_shape2; }
    case 3u: { return params.in_shape3; }
    case 4u: { return params.in_shape4; }
    case 5u: { return params.in_shape5; }
    case 6u: { return params.in_shape6; }
    case 7u: { return params.in_shape7; }
    default: { return 1u; }
  }
}

fn get_in_stride(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.in_stride0; }
    case 1u: { return params.in_stride1; }
    case 2u: { return params.in_stride2; }
    case 3u: { return params.in_stride3; }
    case 4u: { return params.in_stride4; }
    case 5u: { return params.in_stride5; }
    case 6u: { return params.in_stride6; }
    case 7u: { return params.in_stride7; }
    default: { return 0u; }
  }
}

fn get_out_stride(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.out_stride0; }
    case 1u: { return params.out_stride1; }
    case 2u: { return params.out_stride2; }
    case 3u: { return params.out_stride3; }
    case 4u: { return params.out_stride4; }
    case 5u: { return params.out_stride5; }
    case 6u: { return params.out_stride6; }
    case 7u: { return params.out_stride7; }
    default: { return 1u; }
  }
}

fn get_axes_mask(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.axes_mask0; }
    case 1u: { return params.axes_mask1; }
    case 2u: { return params.axes_mask2; }
    case 3u: { return params.axes_mask3; }
    case 4u: { return params.axes_mask4; }
    case 5u: { return params.axes_mask5; }
    case 6u: { return params.axes_mask6; }
    case 7u: { return params.axes_mask7; }
    default: { return 0u; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let out_idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (out_idx >= params.num_out_elements) { return; }

  // 1. Decompose out_idx into unreduced coordinates and compute base_in_idx
  var temp_out = out_idx;
  var base_in_idx = 0u;
  var out_dim_idx = 0u;

  for (var i = 0u; i < params.in_rank; i = i + 1u) {
    if (get_axes_mask(i) == 0u) {
      let out_stride = max(get_out_stride(out_dim_idx), 1u);
      let coord = temp_out / out_stride;
      temp_out = temp_out % out_stride;
      base_in_idx = base_in_idx + coord * get_in_stride(i);
      out_dim_idx = out_dim_idx + 1u;
    }
  }

  // 2. Iterate over the multi-axis reduction space in a single fused loop
  var sum_val = 0.0;
  for (var r = 0u; r < params.reduction_size; r = r + 1u) {
    var temp_r = r;
    var red_in_offset = 0u;
    for (var i = 0u; i < params.in_rank; i = i + 1u) {
      if (get_axes_mask(i) == 1u) {
        let dim_size = max(get_in_shape(i), 1u);
        let coord = temp_r % dim_size;
        temp_r = temp_r / dim_size;
        red_in_offset = red_in_offset + coord * get_in_stride(i);
      }
    }
    sum_val = sum_val + input[base_in_idx + red_in_offset];
  }

  output[out_idx] = sum_val;
}
`;
