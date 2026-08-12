export const COL2IM_WGSL = `
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
@group(0) @binding(1) var<storage, read> grad_x_col: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let num_elements = params.N * params.C * params.H * params.W;
  if (idx >= num_elements) { return; }

  var temp = idx;
  let w = temp % params.W;
  temp = temp / params.W;
  let h = temp % params.H;
  temp = temp / params.H;
  let c = temp % params.C;
  let n = temp / params.C;

  var val = 0.0;
  
  for (var k_h = 0u; k_h < params.K_h; k_h = k_h + 1u) {
    let h_plus_pad = h + params.padding;
    if (h_plus_pad >= k_h) {
      let h_rem = h_plus_pad - k_h;
      if (h_rem % params.stride == 0u) {
        let h_out = h_rem / params.stride;
        if (h_out < params.H_out) {
          
          for (var k_w = 0u; k_w < params.K_w; k_w = k_w + 1u) {
            let w_plus_pad = w + params.padding;
            if (w_plus_pad >= k_w) {
              let w_rem = w_plus_pad - k_w;
              if (w_rem % params.stride == 0u) {
                let w_out = w_rem / params.stride;
                if (w_out < params.W_out) {
                  let n_out = n;
                  let hw_out = h_out * params.W_out + w_out;
                  let c_kw_kh = (c * params.K_h + k_h) * params.K_w + k_w;
                  
                  let col_idx = (n_out * (params.H_out * params.W_out) + hw_out) * (params.C * params.K_h * params.K_w) + c_kw_kh;
                  val = val + grad_x_col[col_idx];
                }
              }
            }
          }
          
        }
      }
    }
  }

  grad_x[idx] = val;
}
`;
