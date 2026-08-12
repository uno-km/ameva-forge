export const FILL_WGSL = `
struct Params {
  numElements: u32,
  value: f32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.numElements;
  let idx = global_id.x;
  if (idx >= num_elements) {
    return;
  }
  output[idx] = params.value;
}
`;
