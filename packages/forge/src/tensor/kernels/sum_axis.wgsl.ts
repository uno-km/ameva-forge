export const SUM_AXIS_WGSL = `
struct Params {
  M: u32,
  N: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let col = global_id.x;
  if (col >= params.N) {
    return;
  }
  
  var sum = 0.0;
  for (var row = 0u; row < params.M; row = row + 1u) {
    sum += input[row * params.N + col];
  }
  
  output[col] = sum;
}
`;
