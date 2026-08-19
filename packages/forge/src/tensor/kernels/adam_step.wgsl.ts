/**
 * 파일 생성일: 2026-08-19
 * AMEVA-Forge Release 2.0 / SCRUM-241: Fused WebGPU Native Adam Step Kernel
 *
 * WHAT: Adam Optimizer의 1차 모멘트(m), 2차 모멘트(v), 편향 보정 및 파라미터 업데이트를 단일 패스로 수행하는 융합 WGSL 커널입니다.
 * WHY: VRAM 왕복 및 CPU readback 없이 GPU 상에서 거대 모델의 Adam 파인튜닝을 100% 네이티브로 가속하기 위함입니다.
 * HOW: m = beta1*m + (1-beta1)*g, v = beta2*v + (1-beta2)*g^2, m_hat = m / (1-beta1^t), v_hat = v / (1-beta2^t),
 *      param = param - lr * m_hat / (sqrt(v_hat) + eps)
 */

export const ADAM_STEP_WGSL = /* wgsl */ `
struct AdamParams {
  num_elements: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  beta1_power: f32,
  beta2_power: f32,
  weight_decay: f32,
  workgroupsX: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: AdamParams;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> param: array<f32>;

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
  let m_prev = m[idx];
  let v_prev = v[idx];

  let m_curr = params.beta1 * m_prev + (1.0 - params.beta1) * g;
  let v_curr = params.beta2 * v_prev + (1.0 - params.beta2) * g * g;

  m[idx] = m_curr;
  v[idx] = v_curr;

  let denom1 = max(1.0 - params.beta1_power, 1e-12);
  let denom2 = max(1.0 - params.beta2_power, 1e-12);
  let m_hat = m_curr / denom1;
  let v_hat = v_curr / denom2;

  let step_update = params.lr * m_hat / (sqrt(max(v_hat, 0.0)) + max(params.eps, 1e-12));
  
  var p_val = param[idx];
  if (params.weight_decay > 0.0) {
    p_val = p_val * (1.0 - params.lr * params.weight_decay);
  }
  param[idx] = p_val - step_update;
}
`;
