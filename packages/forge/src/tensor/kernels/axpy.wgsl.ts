/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Pure Standard IEEE-754 SGD Update without silent NaN/Inf zeroing
 */
export const AXPY_WGSL = `
/**
 * @struct Params
 * @brief AXPY (param = param - lr * grad) 연산 파라미터 구조체
 */
struct Params {
  numElements: u32,
  lr: f32,
  workgroups_x: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  if (idx >= params.numElements) {
    return;
  }
  
  let g = grad[idx];
  // Standard SGD in-place update (IEEE 754 float32)
  param[idx] = param[idx] - params.lr * g;
}
`;
