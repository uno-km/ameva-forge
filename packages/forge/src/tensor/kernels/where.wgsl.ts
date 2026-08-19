/**
 * 파일 생성: 2026-08-12 12:23:09
 * 수정 내역:
 * - 2026-08-12 12:23:09: Docs: Build Apache-style docs and unify tests (fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
 */
export const WHERE_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  rank: u32,
  pad: u32,
  d_out: array<u32, 8>,
  eff_s_cond: array<u32, 8>,
  eff_s_x: array<u32, 8>,
  eff_s_y: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cond: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  if (idx >= num_elements) {
    return;
  }

  var temp = idx;
  var cond_idx = 0u;
  var x_idx = 0u;
  var y_idx = 0u;

  for (var i: i32 = 7; i >= 0; i = i - 1) {
    let u_i = u32(i);
    let dim_size = params.d_out[u_i];
    let coord = temp % dim_size;
    temp = temp / dim_size;

    cond_idx = cond_idx + coord * params.eff_s_cond[u_i];
    x_idx = x_idx + coord * params.eff_s_x[u_i];
    y_idx = y_idx + coord * params.eff_s_y[u_i];
  }

  if (cond[cond_idx] != 0.0) {
    out[idx] = x[x_idx];
  } else {
    out[idx] = y[y_idx];
  }
}
`;
