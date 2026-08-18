/**
 * 파일 생성일: 2026-08-19
 * AMEVA-Forge Release 2.0 / SCRUM-242: Fused WebGPU Native Momentum SGD Step Kernel
 *
 * WHAT: Momentum SGD의 velocity 갱신 및 파라미터 업데이트를 단일 패스로 수행하는 융합 WGSL 커널입니다.
 * WHY: GPU 상에서 가속된 모멘텀 기울기 하강을 VRAM 인플레이스로 직접 수행하기 위함입니다.
 * HOW: v = momentum * v + grad, param = param - lr * v
 */

export const SGD_MOMENTUM_STEP_WGSL = /* wgsl */ `
struct MomentumParams {
  num_elements: u32,
  lr: f32,
  momentum: f32,
  workgroupsX: u32,
};

@group(0) @binding(0) var<uniform> params: MomentumParams;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(3) var<storage, read_write> param: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroupsX) * 64u + thread_id;

  if (idx >= params.num_elements) {
    return;
  }

  let g = grad[idx];
  let v_prev = velocity[idx];

  let v_curr = params.momentum * v_prev + g;
  velocity[idx] = v_curr;

  param[idx] = param[idx] - params.lr * v_curr;
}
`;
