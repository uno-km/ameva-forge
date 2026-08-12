export const IM2COL_WGSL = `
struct Params {
  N: u32,
  C: u32,
  H: u32,
  W: u32,
  K_h: u32,
  K_w: u32,
  stride: u32,
  padding: u32,
  H_out: u32,
  W_out: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let num_elements = params.N * params.H_out * params.W_out * params.C * params.K_h * params.K_w;
  if (idx >= num_elements) { return; }

  var temp = idx;
  let c_kw_kh = temp % (params.C * params.K_h * params.K_w);
  temp = temp / (params.C * params.K_h * params.K_w);
  let h_out_w_out = temp % (params.H_out * params.W_out);
  temp = temp / (params.H_out * params.W_out);
  let n = temp % params.N;

  let k_w = c_kw_kh % params.K_w;
  let k_h = (c_kw_kh / params.K_w) % params.K_h;
  let c = c_kw_kh / (params.K_w * params.K_h);

  let w_out = h_out_w_out % params.W_out;
  let h_out = h_out_w_out / params.W_out;

  let h_in = i32(h_out * params.stride) - i32(params.padding) + i32(k_h);
  let w_in = i32(w_out * params.stride) - i32(params.padding) + i32(k_w);

  if (h_in >= 0 && h_in < i32(params.H) && w_in >= 0 && w_in < i32(params.W)) {
    let in_idx = ((n * params.C + c) * params.H + u32(h_in)) * params.W + u32(w_in);
    output[idx] = input[in_idx];
  } else {
    output[idx] = 0.0;
  }
}
`;
