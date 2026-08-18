/**
 * 파일 생성일: 2026-08-19
 * AMEVA-Forge Release 2.0 / SCRUM-241: Fused WebGPU Native Adam Step Kernel
 *
 * WHAT: Adam Optimizer의 1차 모멘트(m), 2차 모멘트(v), 편향 보정 및 파라미터 업데이트를 단일 패스로 수행하는 융합 WGSL 커널입니다.
 * WHY: VRAM 왕복 및 CPU readback 없이 GPU 상에서 거대 모델의 Adam 파인튜닝을 100% 네이티브로 가속하기 위함입니다.
 * HOW: m = beta1*m + (1-beta1)*g, v = beta2*v + (1-beta2)*g^2, m_hat = m / (1-beta1^t), v_hat = v / (1-beta2^t),
 *      param = param - lr * m_hat / (sqrt(v_hat) + eps)
 */
export declare const ADAM_STEP_WGSL = "\nstruct AdamParams {\n  num_elements: u32,\n  lr: f32,\n  beta1: f32,\n  beta2: f32,\n  eps: f32,\n  beta1_power: f32,\n  beta2_power: f32,\n  workgroupsX: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: AdamParams;\n@group(0) @binding(1) var<storage, read> grad: array<f32>;\n@group(0) @binding(2) var<storage, read_write> m: array<f32>;\n@group(0) @binding(3) var<storage, read_write> v: array<f32>;\n@group(0) @binding(4) var<storage, read_write> param: array<f32>;\n\n@compute @workgroup_size(64, 1, 1)\nfn main(\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n  @builtin(workgroup_id) workgroup_id: vec3<u32>\n) {\n  let thread_id = local_id.x;\n  let idx = (workgroup_id.x + workgroup_id.y * params.workgroupsX) * 64u + thread_id;\n\n  if (idx >= params.num_elements) {\n    return;\n  }\n\n  let g = grad[idx];\n  let m_prev = m[idx];\n  let v_prev = v[idx];\n\n  let m_curr = params.beta1 * m_prev + (1.0 - params.beta1) * g;\n  let v_curr = params.beta2 * v_prev + (1.0 - params.beta2) * g * g;\n\n  m[idx] = m_curr;\n  v[idx] = v_curr;\n\n  let m_hat = m_curr / (1.0 - params.beta1_power);\n  let v_hat = v_curr / (1.0 - params.beta2_power);\n\n  let step_update = params.lr * m_hat / (sqrt(v_hat) + params.eps);\n  param[idx] = param[idx] - step_update;\n}\n";
