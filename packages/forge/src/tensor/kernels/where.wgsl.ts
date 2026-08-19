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
  pad0: u32,
  dim0: u32, dim1: u32, dim2: u32, dim3: u32,
  dim4: u32, dim5: u32, dim6: u32, dim7: u32,
  stride_cond0: u32, stride_cond1: u32, stride_cond2: u32, stride_cond3: u32,
  stride_cond4: u32, stride_cond5: u32, stride_cond6: u32, stride_cond7: u32,
  stride_x0: u32, stride_x1: u32, stride_x2: u32, stride_x3: u32,
  stride_x4: u32, stride_x5: u32, stride_x6: u32, stride_x7: u32,
  stride_y0: u32, stride_y1: u32, stride_y2: u32, stride_y3: u32,
  stride_y4: u32, stride_y5: u32, stride_y6: u32, stride_y7: u32,
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
  let c7 = temp % params.dim7; temp = temp / params.dim7;
  let c6 = temp % params.dim6; temp = temp / params.dim6;
  let c5 = temp % params.dim5; temp = temp / params.dim5;
  let c4 = temp % params.dim4; temp = temp / params.dim4;
  let c3 = temp % params.dim3; temp = temp / params.dim3;
  let c2 = temp % params.dim2; temp = temp / params.dim2;
  let c1 = temp % params.dim1; temp = temp / params.dim1;
  let c0 = temp;

  let cond_idx = c0 * params.stride_cond0 + c1 * params.stride_cond1 + c2 * params.stride_cond2 + c3 * params.stride_cond3 +
                 c4 * params.stride_cond4 + c5 * params.stride_cond5 + c6 * params.stride_cond6 + c7 * params.stride_cond7;
  let x_idx    = c0 * params.stride_x0 + c1 * params.stride_x1 + c2 * params.stride_x2 + c3 * params.stride_x3 +
                 c4 * params.stride_x4 + c5 * params.stride_x5 + c6 * params.stride_x6 + c7 * params.stride_x7;
  let y_idx    = c0 * params.stride_y0 + c1 * params.stride_y1 + c2 * params.stride_y2 + c3 * params.stride_y3 +
                 c4 * params.stride_y4 + c5 * params.stride_y5 + c6 * params.stride_y6 + c7 * params.stride_y7;

  if (cond[cond_idx] != 0.0) {
    out[idx] = x[x_idx];
  } else {
    out[idx] = y[y_idx];
  }
}
`;
