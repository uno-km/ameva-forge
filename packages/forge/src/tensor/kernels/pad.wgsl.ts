/**
 * 생성일: 2026-08-12T12:23:09+09:00
 * 수정 이력:
 * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 */
export const PAD_WGSL = `
struct Params {
  num_elements: u32,
  rank: u32,
  pad_val: f32,
  workgroups_x: u32,
  in_stride0: u32, in_stride1: u32, in_stride2: u32, in_stride3: u32,
  in_stride4: u32, in_stride5: u32, in_stride6: u32, in_stride7: u32,
  out_stride0: u32, out_stride1: u32, out_stride2: u32, out_stride3: u32,
  out_stride4: u32, out_stride5: u32, out_stride6: u32, out_stride7: u32,
  pad_before0: u32, pad_before1: u32, pad_before2: u32, pad_before3: u32,
  pad_before4: u32, pad_before5: u32, pad_before6: u32, pad_before7: u32,
  in_shape0: u32, in_shape1: u32, in_shape2: u32, in_shape3: u32,
  in_shape4: u32, in_shape5: u32, in_shape6: u32, in_shape7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn get_out_stride(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.out_stride0; }
    case 1u: { return params.out_stride1; }
    case 2u: { return params.out_stride2; }
    case 3u: { return params.out_stride3; }
    case 4u: { return params.out_stride4; }
    case 5u: { return params.out_stride5; }
    case 6u: { return params.out_stride6; }
    default: { return params.out_stride7; }
  }
}

fn get_in_stride(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.in_stride0; }
    case 1u: { return params.in_stride1; }
    case 2u: { return params.in_stride2; }
    case 3u: { return params.in_stride3; }
    case 4u: { return params.in_stride4; }
    case 5u: { return params.in_stride5; }
    case 6u: { return params.in_stride6; }
    default: { return params.in_stride7; }
  }
}

fn get_pad_before(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.pad_before0; }
    case 1u: { return params.pad_before1; }
    case 2u: { return params.pad_before2; }
    case 3u: { return params.pad_before3; }
    case 4u: { return params.pad_before4; }
    case 5u: { return params.pad_before5; }
    case 6u: { return params.pad_before6; }
    default: { return params.pad_before7; }
  }
}

fn get_in_shape(i: u32) -> u32 {
  switch (i) {
    case 0u: { return params.in_shape0; }
    case 1u: { return params.in_shape1; }
    case 2u: { return params.in_shape2; }
    case 3u: { return params.in_shape3; }
    case 4u: { return params.in_shape4; }
    case 5u: { return params.in_shape5; }
    case 6u: { return params.in_shape6; }
    default: { return params.in_shape7; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;
  var in_bounds = true;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let out_stride = max(get_out_stride(i), 1u);
    let coord = temp / out_stride;
    temp = temp % out_stride;

    let pad_b = get_pad_before(i);
    let in_s = get_in_shape(i);
    if (coord < pad_b || coord >= pad_b + in_s) {
      in_bounds = false;
      break;
    }

    let in_coord = coord - pad_b;
    in_idx = in_idx + in_coord * get_in_stride(i);
  }

  if (in_bounds) {
    output[idx] = input[in_idx];
  } else {
    output[idx] = params.pad_val;
  }
}
`;
