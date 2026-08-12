export const GATHER_WGSL = `
struct Params {
  num_elements: u32,
  dim: u32,
  rank: u32,
  _pad: u32,
  x_strides: array<u32, 8>,
  out_strides: array<u32, 8>,
  x_shape: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.out_strides[i];
    temp = temp % params.out_strides[i];
    
    if (i == params.dim) {
      let idx_val = u32(index[idx]);
      in_idx = in_idx + idx_val * params.x_strides[i];
    } else {
      in_idx = in_idx + coord * params.x_strides[i];
    }
  }

  output[idx] = input[in_idx];
}
`;
