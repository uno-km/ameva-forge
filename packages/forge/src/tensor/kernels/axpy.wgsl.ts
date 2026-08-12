export const AXPY_WGSL = `
struct Params {
  numElements: u32,
  lr: f32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.numElements) {
    return;
  }
  param[idx] = param[idx] - params.lr * grad[idx];
}
`;
