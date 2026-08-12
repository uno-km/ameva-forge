export const MUL_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> A : array<f32>;
@group(0) @binding(2) var<storage, read> B : array<f32>;
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  if (index < num_elements) {
    C[index] = A[index] * B[index];
  }
}
`;
