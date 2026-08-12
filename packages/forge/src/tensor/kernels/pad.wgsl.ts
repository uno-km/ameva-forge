export const PAD_WGSL = `
struct Params {
  num_elements: u32,
  rank: u32,
  pad_val: f32,
  _pad: u32,
  in_strides: array<u32, 8>,
  out_strides: array<u32, 8>,
  pad_before: array<u32, 8>,
  in_shape: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.num_elements) { return; }

  var temp = idx;
  var in_idx = 0u;
  var in_bounds = true;

  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.out_strides[i];
    temp = temp % params.out_strides[i];
    
    if (coord < params.pad_before[i] || coord >= params.pad_before[i] + params.in_shape[i]) {
      in_bounds = false;
      break;
    }
    let in_coord = coord - params.pad_before[i];
    in_idx = in_idx + in_coord * params.in_strides[i];
  }

  if (in_bounds) {
    output[idx] = input[in_idx];
  } else {
    output[idx] = params.pad_val;
  }
}
`;
