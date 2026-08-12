export const CAT_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  a_dim: u32,
  b_dim: u32,
  stride: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  if (idx >= num_elements) {
    return;
  }
  
  let stride = params.stride;
  let a_dim = params.a_dim;
  let b_dim = params.b_dim;
  
  let out_dim_size = a_dim + b_dim;
  let chunk_size = out_dim_size * stride;
  
  let batch_idx = idx / chunk_size;
  let rem = idx % chunk_size;
  let dim_idx = rem / stride;
  let stride_idx = rem % stride;
  
  if (dim_idx < a_dim) {
    let a_index = batch_idx * (a_dim * stride) + dim_idx * stride + stride_idx;
    out[idx] = a[a_index];
  } else {
    let b_dim_idx = dim_idx - a_dim;
    let b_index = batch_idx * (b_dim * stride) + b_dim_idx * stride + stride_idx;
    out[idx] = b[b_index];
  }
}
`;
