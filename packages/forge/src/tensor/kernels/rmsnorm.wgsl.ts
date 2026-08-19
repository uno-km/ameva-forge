/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-220 RMSNorm WGSL Kernel
 *
 * WHAT: Root Mean Square Normalization (RMSNorm) WGSL 셰이더입니다.
 * WHY: LayerNorm 대비 평균 계산 오버헤드를 제거하여 연산 속도를 20~30% 단축하고, 수치적 안정성을 제공하기 위해 존재합니다.
 * HOW: 각 토큰 벡터의 제곱합을 계산하여 RMS 값을 구한 후, 스케일 파라미터(gamma)를 곱하여 정규화된 텐서를 산출합니다.
 */

export const RMSNORM_WGSL = `
struct Params {
  num_tokens: u32,  // 총 토큰 수 (Batch * SeqLen)
  dim: u32,         // 은닉 차원 (Hidden Dim, 예: 2048, 4096)
  eps: f32,         // 수치 안정화 epsilon (예: 1e-5, 1e-6)
  has_gamma: u32,   // 1: gamma 스케일 적용, 0: 생략
  workgroupsX: u32, // 2D 디스패치 X축 워크그룹 수
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> s_sum_sq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let token_idx = workgroup_id.x + workgroup_id.y * params.workgroupsX;

  if (token_idx >= params.num_tokens) {
    return;
  }

  let token_offset = token_idx * params.dim;

  // 1. 스레드별 제곱합 계산
  var local_sum_sq: f32 = 0.0;
  for (var i: u32 = thread_id; i < params.dim; i = i + 256u) {
    let val = x[token_offset + i];
    local_sum_sq = local_sum_sq + val * val;
  }
  s_sum_sq[thread_id] = local_sum_sq;

  workgroupBarrier();

  // 2. 워크그룹 트리 리덕션 (Tree Reduction)
  for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
    if (thread_id < stride) {
      s_sum_sq[thread_id] = s_sum_sq[thread_id] + s_sum_sq[thread_id + stride];
    }
    workgroupBarrier();
  }

  // 3. RMS 스케일 계산: 1.0 / sqrt(mean_sq + eps)
  let mean_sq = s_sum_sq[0] / f32(params.dim);
  let inv_rms = 1.0 / sqrt(mean_sq + params.eps);

  // 4. 정규화 및 Gamma 스케일링
  for (var i: u32 = thread_id; i < params.dim; i = i + 256u) {
    var val = x[token_offset + i] * inv_rms;
    if (params.has_gamma == 1u) {
      val = val * gamma[i];
    }
    out[token_offset + i] = val;
  }
}
`;
