export const BATCHED_MATMUL_WGSL = `
struct Params {
  B: u32,
  M: u32,
  N: u32,
  K: u32,
  strideA: u32,
  strideB: u32,
  strideC: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x;
  let row = global_id.y;
  let batch = global_id.z;

  if (row >= params.M || col >= params.N || batch >= params.B) {
    return;
  }

  let a_offset = batch * params.strideA + row * params.K;
  let b_offset = batch * params.strideB + col;
  let c_offset = batch * params.strideC + row * params.N + col;

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[a_offset + k] * b[b_offset + k * params.N];
  }

  c[c_offset] = sum;
}
`;
