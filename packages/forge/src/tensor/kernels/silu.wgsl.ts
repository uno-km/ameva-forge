/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-311 SiLU (Swish) Fused Activation WGSL Compute Kernel
 *
 * WHAT: 디퓨전(Stable Diffusion UNet) 신경망의 표준 활성화 함수인 SiLU(x * sigmoid(x))의 GPU 순전파 및 역전파 WGSL 커널입니다.
 * WHY: Stable Diffusion UNet ResNet 블록 전반에 수십 번 적용되는 SiLU를 메모리 복사 없이 초고속 In-place / Streaming GPU 커널로 실행하기 위해 존재합니다.
 * HOW: 수치 안정성을 위해 sigmoid(x) = 1.0 / (1.0 + exp(-clamp(x, -88.0, 88.0)))을 적용하고 x * sigmoid(x)를 계산합니다.
 *      Uniform 레이아웃은 WebGPU 16바이트 정렬 규격을 100% 준수합니다 (4 x 4바이트 = 16바이트).
 */

export const SILU_WGSL = `
struct Params {
  num_elements: u32,  // 총 계산 원소 개수
  workgroups_x: u32,  // 2D 디스패치 X축 워크그룹 크기
  pad0: u32,          // 16바이트 유니폼 정렬용 패딩 1
  pad1: u32,          // 16바이트 유니폼 정렬용 패딩 2
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn stable_sigmoid(x: f32) -> f32 {
  let clamped_x = clamp(x, -88.0, 88.0);
  return 1.0 / (1.0 + exp(-clamped_x));
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;

  if (idx >= params.num_elements) {
    return;
  }

  let x = input[idx];
  let sig = stable_sigmoid(x);
  output[idx] = x * sig;
}
`;

export const SILU_BACKWARD_WGSL = `
struct Params {
  num_elements: u32,
  workgroups_x: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;

fn stable_sigmoid(x: f32) -> f32 {
  let clamped_x = clamp(x, -88.0, 88.0);
  return 1.0 / (1.0 + exp(-clamped_x));
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;

  if (idx >= params.num_elements) {
    return;
  }

  let x = input[idx];
  let sig = stable_sigmoid(x);
  let d_act = sig * (1.0 + x * (1.0 - sig));
  grad_input[idx] = grad_output[idx] * d_act;
}
`;
