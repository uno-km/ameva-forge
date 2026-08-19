/**
 * Native WebGPU Slice Backward Compute Kernel
 * Accumulates gradients from sliced output back into the full input gradient tensor.
 * Fully compliant with W3C WebGPU 16-byte uniform alignment rules using scalar fields.
 */
export const SLICE_BACKWARD_WGSL = `
struct Params {
  num_elements: u32,
  rank: u32,
  workgroups_x: u32,
  pad: u32,
  start0: u32, start1: u32, start2: u32, start3: u32,
  start4: u32, start5: u32, start6: u32, start7: u32,
  step0: u32, step1: u32, step2: u32, step3: u32,
  step4: u32, step5: u32, step6: u32, step7: u32,
  in_stride0: u32, in_stride1: u32, in_stride2: u32, in_stride3: u32,
  in_stride4: u32, in_stride5: u32, in_stride6: u32, in_stride7: u32,
  out_stride0: u32, out_stride1: u32, out_stride2: u32, out_stride3: u32,
  out_stride4: u32, out_stride5: u32, out_stride6: u32, out_stride7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;

fn get_start(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.start0; }
    case 1u: { return params.start1; }
    case 2u: { return params.start2; }
    case 3u: { return params.start3; }
    case 4u: { return params.start4; }
    case 5u: { return params.start5; }
    case 6u: { return params.start6; }
    case 7u: { return params.start7; }
    default: { return 0u; }
  }
}

fn get_step(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.step0; }
    case 1u: { return params.step1; }
    case 2u: { return params.step2; }
    case 3u: { return params.step3; }
    case 4u: { return params.step4; }
    case 5u: { return params.step5; }
    case 6u: { return params.step6; }
    case 7u: { return params.step7; }
    default: { return 1u; }
  }
}

fn get_in_stride(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.in_stride0; }
    case 1u: { return params.in_stride1; }
    case 2u: { return params.in_stride2; }
    case 3u: { return params.in_stride3; }
    case 4u: { return params.in_stride4; }
    case 5u: { return params.in_stride5; }
    case 6u: { return params.in_stride6; }
    case 7u: { return params.in_stride7; }
    default: { return 0u; }
  }
}

fn get_out_stride(i: u32) -> u32 {
  switch(i) {
    case 0u: { return params.out_stride0; }
    case 1u: { return params.out_stride1; }
    case 2u: { return params.out_stride2; }
    case 3u: { return params.out_stride3; }
    case 4u: { return params.out_stride4; }
    case 5u: { return params.out_stride5; }
    case 6u: { return params.out_stride6; }
    case 7u: { return params.out_stride7; }
    default: { return 1u; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;
  for (var i = 0u; i < params.rank; i = i + 1u) {
    let out_stride = max(get_out_stride(i), 1u);
    let coord = temp / out_stride;
    temp = temp % out_stride;

    let in_coord = get_start(i) + coord * get_step(i);
    in_idx = in_idx + in_coord * get_in_stride(i);
  }
  grad_x[in_idx] = grad_output[idx];
}
`;
