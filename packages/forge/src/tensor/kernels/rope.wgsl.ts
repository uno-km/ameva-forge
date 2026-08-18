/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-219 Rotary Position Embedding (RoPE) WGSL Kernel
 *
 * WHAT: LLaMA / Mistral / Gemma 등 현대 LLM의 핵심 위치 인코딩인 RoPE(Rotary Position Embedding) WGSL 셰이더입니다.
 * WHY: 입력 시퀀스의 상대적 위치 정보를 복소수 회전 행렬 형태로 Query 및 Key 텐서에 인플레이스 주입하기 위해 존재합니다.
 * HOW: 각 토큰 위치(pos)와 헤드 차원 페어 인덱스(k)에 대해 주파수 theta = base^(-2k/d)를 계산하고,
 *      cos/sin 삼각함수를 적용하여 2D 평면 회전 변환을 단일 GPU 패스로 수행합니다.
 */

export const ROPE_WGSL = `
struct Params {
  B: u32,             // 총 배치 수
  H: u32,             // 헤드 수
  N: u32,             // 시퀀스 길이
  d: u32,             // 헤드 차원 (반드시 짝수, 예: 64, 128)
  base_freq: f32,     // 기본 주파수 (예: 10000.0 또는 500000.0)
  offset_pos: u32,    // KV 캐시 오프셋 위치 (Prefill / Decode 단계별 시작 토큰 인덱스)
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let pair_idx = local_id.x; // 0 .. (d/2 - 1)
  let token_idx = workgroup_id.x; // 0 .. N-1
  let head_idx = workgroup_id.y;  // 0 .. H-1
  let batch_idx = workgroup_id.z; // 0 .. B-1

  let half_d = params.d / 2u;
  if (pair_idx >= half_d || token_idx >= params.N || head_idx >= params.H || batch_idx >= params.B) {
    return;
  }

  let pos = f32(token_idx + params.offset_pos);
  let freq_exponent = -2.0 * f32(pair_idx) / f32(params.d);
  let theta = pow(params.base_freq, freq_exponent) * pos;

  let cos_theta = cos(theta);
  let sin_theta = sin(theta);

  let tensor_offset = ((batch_idx * params.H + head_idx) * params.N + token_idx) * params.d;
  let idx0 = tensor_offset + pair_idx * 2u;
  let idx1 = tensor_offset + pair_idx * 2u + 1u;

  let v0 = x[idx0];
  let v1 = x[idx1];

  // 2D 회전 변환
  out[idx0] = v0 * cos_theta - v1 * sin_theta;
  out[idx1] = v1 * cos_theta + v0 * sin_theta;
}
`;
