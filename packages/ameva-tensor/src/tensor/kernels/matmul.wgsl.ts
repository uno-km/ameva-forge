export const MATMUL_WGSL = `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  offsetY: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.y + params.offsetY;
  let col = global_id.x;

  if (row >= params.M || col >= params.N) {
    return;
  }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  c[row * params.N + col] = sum;
}
`;
