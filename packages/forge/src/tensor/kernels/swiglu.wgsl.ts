/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-221 SwiGLU Fused Activation WGSL Kernel
 *
 * WHAT: Swish Gated Linear Unit (SwiGLU) 융합 활성화 함수 WGSL 셰이더입니다.
 * WHY: LLaMA 및 Gemma 등의 FFN 블록에서 Gate Projection(x)과 Up Projection(y)의 원소별 Swish 게이팅을
 *      중간 메모리 왕복 없이 단일 커널로 초고속 처리하기 위해 존재합니다.
 * HOW: Swish(x) = x * sigmoid(x) = x / (1.0 + exp(-x)) 연산 후 y와 원소별 곱셈을 수행합니다.
 */

export const SWIGLU_WGSL = `
struct Params {
  num_elements: u32,  // 총 원소 개수
  workgroupsX: u32,   // 2D 디스패치 X 크기
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>; // Gate projection (x)
@group(0) @binding(2) var<storage, read> up: array<f32>;   // Up projection (y)
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // SwiGLU output

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroupsX) * 64u + local_id.x;

  if (idx >= params.num_elements) {
    return;
  }

  let x = gate[idx];
  let y = up[idx];

  // Swish(x) = x / (1.0 + exp(-x))
  let swish_x = x / (1.0 + exp(-x));
  out[idx] = swish_x * y;
}
`;
