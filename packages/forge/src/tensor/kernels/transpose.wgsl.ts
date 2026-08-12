export const TRANSPOSE_WGSL = `
struct Params {
  M: u32,
  N: u32,
  B: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;
  let batch = global_id.z;
  
  if (row < params.M && col < params.N && batch < params.B) {
    let in_idx = batch * (params.M * params.N) + row * params.N + col;
    let out_idx = batch * (params.M * params.N) + col * params.M + row;
    out[out_idx] = input[in_idx];
  }
}
`;
