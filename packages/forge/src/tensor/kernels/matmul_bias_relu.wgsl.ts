/**
 * AMEVA-Forge Fused Linear Kernel: MatMul + BiasAdd + ReLU
 * Computes C = ReLU(A @ B + Bias) in a single GPU compute pass.
 */
export const MATMUL_BIAS_RELU_WGSL = `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  offsetY: u32,
  has_bias: u32,
  has_relu: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x + global_id.z * 65535u * 8u;
  let row = global_id.y + params.offsetY;

  if (row >= params.M || col >= params.N) {
    return;
  }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  if (params.has_bias == 1u) {
    sum = sum + bias[col];
  }

  if (params.has_relu == 1u) {
    sum = max(sum, 0.0);
  }

  c[row * params.N + col] = sum;
}
`;
