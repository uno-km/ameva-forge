export const SUM_WGSL = `
struct Params {
  numElements: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let gid = global_id.x;
  let lid = local_id.x;
  let wid = workgroup_id.x;
  
  if (gid < params.numElements) {
    shared[lid] = input[gid];
  } else {
    shared[lid] = 0.0;
  }
  
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      shared[lid] += shared[lid + s];
    }
    workgroupBarrier();
  }
  
  if (lid == 0u) {
    output[wid] = shared[0];
  }
}
`;
