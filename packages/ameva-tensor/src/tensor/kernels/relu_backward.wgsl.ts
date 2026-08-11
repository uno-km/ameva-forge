export const RELU_BACKWARD_WGSL = `
struct Params {
  size: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read> gradOutput : array<f32>;
@group(0) @binding(3) var<storage, read_write> gradInput : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index < params.size) {
    if (X[index] > 0.0) {
      gradInput[index] = gradOutput[index];
    } else {
      gradInput[index] = 0.0;
    }
  }
}
`;
