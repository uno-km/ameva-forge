/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Feat: Full 8D Multi-Dimensional Stride Broadcasting Decoder
 */
export const MUL_WGSL = `
struct Params {
  size: u32,
  workgroups_x: u32,
  rank: u32,
  pad0: u32,
  dim0: u32, dim1: u32, dim2: u32, dim3: u32,
  dim4: u32, dim5: u32, dim6: u32, dim7: u32,
  stride_a0: u32, stride_a1: u32, stride_a2: u32, stride_a3: u32,
  stride_a4: u32, stride_a5: u32, stride_a6: u32, stride_a7: u32,
  stride_b0: u32, stride_b1: u32, stride_b2: u32, stride_b3: u32,
  stride_b4: u32, stride_b5: u32, stride_b6: u32, stride_b7: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    var temp = idx;
    let c7 = temp % params.dim7; temp = temp / params.dim7;
    let c6 = temp % params.dim6; temp = temp / params.dim6;
    let c5 = temp % params.dim5; temp = temp / params.dim5;
    let c4 = temp % params.dim4; temp = temp / params.dim4;
    let c3 = temp % params.dim3; temp = temp / params.dim3;
    let c2 = temp % params.dim2; temp = temp / params.dim2;
    let c1 = temp % params.dim1; temp = temp / params.dim1;
    let c0 = temp;

    let idx_a = c0 * params.stride_a0 + c1 * params.stride_a1 + c2 * params.stride_a2 + c3 * params.stride_a3 +
                c4 * params.stride_a4 + c5 * params.stride_a5 + c6 * params.stride_a6 + c7 * params.stride_a7;
    let idx_b = c0 * params.stride_b0 + c1 * params.stride_b1 + c2 * params.stride_b2 + c3 * params.stride_b3 +
                c4 * params.stride_b4 + c5 * params.stride_b5 + c6 * params.stride_b6 + c7 * params.stride_b7;

    out[idx] = a[idx_a] * b[idx_b];
  }
}
`;
