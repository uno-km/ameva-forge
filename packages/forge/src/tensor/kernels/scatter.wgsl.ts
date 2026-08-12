export const SCATTER_WGSL = `
struct Params {
  num_elements: u32,
  dim: u32,
  rank: u32,
  _pad: u32,
  x_strides: array<u32, 8>,
  idx_strides: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> index: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var out_idx = 0u;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.idx_strides[i];
    temp = temp % params.idx_strides[i];
    
    if (i == params.dim) {
      let idx_val = u32(index[idx]);
      out_idx = out_idx + idx_val * params.x_strides[i];
    } else {
      out_idx = out_idx + coord * params.x_strides[i];
    }
  }

  // Not strictly atomic, but for simple scatter where indices are unique it's fine.
  output[out_idx] = src[idx];
}
`;
