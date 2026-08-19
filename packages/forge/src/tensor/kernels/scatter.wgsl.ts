/**
 * 생성일: 2026-08-12T12:23:09+09:00
 * 수정 이력:
 * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 */
export const SCATTER_WGSL = `
struct Params {
  num_elements: u32,
  dim: u32,
  rank: u32,
  workgroups_x: u32,
  x_stride0: u32, x_stride1: u32, x_stride2: u32, x_stride3: u32,
  x_stride4: u32, x_stride5: u32, x_stride6: u32, x_stride7: u32,
  idx_stride0: u32, idx_stride1: u32, idx_stride2: u32, idx_stride3: u32,
  idx_stride4: u32, idx_stride5: u32, idx_stride6: u32, idx_stride7: u32,
  x_shape0: u32, x_shape1: u32, x_shape2: u32, x_shape3: u32,
  x_shape4: u32, x_shape5: u32, x_shape6: u32, x_shape7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn get_idx_stride(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.idx_stride0; }
    case 1u: { return params.idx_stride1; }
    case 2u: { return params.idx_stride2; }
    case 3u: { return params.idx_stride3; }
    case 4u: { return params.idx_stride4; }
    case 5u: { return params.idx_stride5; }
    case 6u: { return params.idx_stride6; }
    default: { return params.idx_stride7; }
  }
}

fn get_x_stride(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.x_stride0; }
    case 1u: { return params.x_stride1; }
    case 2u: { return params.x_stride2; }
    case 3u: { return params.x_stride3; }
    case 4u: { return params.x_stride4; }
    case 5u: { return params.x_stride5; }
    case 6u: { return params.x_stride6; }
    default: { return params.x_stride7; }
  }
}

fn get_x_shape(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.x_shape0; }
    case 1u: { return params.x_shape1; }
    case 2u: { return params.x_shape2; }
    case 3u: { return params.x_shape3; }
    case 4u: { return params.x_shape4; }
    case 5u: { return params.x_shape5; }
    case 6u: { return params.x_shape6; }
    default: { return params.x_shape7; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var out_idx = 0u;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let idx_stride = max(get_idx_stride(i), 1u);
    let coord = temp / idx_stride;
    temp = temp % idx_stride;
    
    if (i == params.dim) {
      let raw_val = index[idx];
      if (raw_val != raw_val) {
        return;
      }
      let dim_size = i32(get_x_shape(i));
      var signed_idx = i32(round(raw_val));
      if (signed_idx < 0) {
        signed_idx = signed_idx + dim_size;
      }
      if (signed_idx < 0 || signed_idx >= dim_size) {
        return;
      }
      let valid_idx = u32(signed_idx);
      out_idx = out_idx + valid_idx * get_x_stride(i);
    } else {
      out_idx = out_idx + coord * get_x_stride(i);
    }
  }

  output[out_idx] = src[idx];
}
`;
