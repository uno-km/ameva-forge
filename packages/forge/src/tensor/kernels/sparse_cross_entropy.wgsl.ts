/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/src/tensor/kernels/sparse_cross_entropy.wgsl.ts
 * Type: WebGPU WGSL Compute Kernel (Fused Sparse Cross-Entropy Forward)
 * Created: 2026-08-19T01:00:00+09:00
 * ============================================================================
 * WHAT:
 *   [N, C] 크기의 Logits 텐서와 [N] 크기의 정수 Target 텐서를 받아
 *   Dense One-Hot 행렬 생성 없이 VRAM O(N)으로 직접 Cross-Entropy Loss를 계산하는 융합 커널입니다.
 * WHY:
 *   LLM과 같이 어휘집 크기(C=32k~128k)가 큰 모델에서 Dense One-Hot 할당으로 인한 VRAM OOM을 100% 제거하기 위함입니다.
 * HOW:
 *   1개 워크그룹(256 스레드)이 1개 배치 샘플을 전담하여, 공유 메모리 2단계 병렬 트리 리덕션으로
 *   Max 값과 Log-Sum-Exp를 계산한 뒤, 정수 타겟 인덱스의 NLL Loss를 직접 산출합니다.
 */

export const SPARSE_CROSS_ENTROPY_WGSL = /* wgsl */ `
struct Params {
  num_samples: u32,
  num_classes: u32,
  ignore_index: i32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<u32>;
@group(0) @binding(3) var<storage, read_write> loss: array<f32>;

var<workgroup> s_max: array<f32, 256>;
var<workgroup> s_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let sample_idx = workgroup_id.x + workgroup_id.y * 65535u;

  if (sample_idx >= params.num_samples) {
    return;
  }

  let row_offset = sample_idx * params.num_classes;

  // 1. 최대값(Max) 탐색 (수치 안정성 확보)
  var local_max: f32 = -3.402823e+38;
  for (var c: u32 = thread_id; c < params.num_classes; c = c + 256u) {
    let val = logits[row_offset + c];
    local_max = max(local_max, val);
  }
  s_max[thread_id] = local_max;

  workgroupBarrier();

  for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
    if (thread_id < stride) {
      s_max[thread_id] = max(s_max[thread_id], s_max[thread_id + stride]);
    }
    workgroupBarrier();
  }

  let max_val = s_max[0];

  // 2. Sum of Exponentials 계산
  var local_sum: f32 = 0.0;
  for (var c: u32 = thread_id; c < params.num_classes; c = c + 256u) {
    let val = logits[row_offset + c];
    local_sum = local_sum + exp(val - max_val);
  }
  s_sum[thread_id] = local_sum;

  workgroupBarrier();

  for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
    if (thread_id < stride) {
      s_sum[thread_id] = s_sum[thread_id] + s_sum[thread_id + stride];
    }
    workgroupBarrier();
  }

  let sum_exp = s_sum[0];

  // 3. Thread 0이 NLL Loss 계산 및 출력 버퍼에 기록
  if (thread_id == 0u) {
    let raw_target = targets[sample_idx];
    let target_idx = i32(raw_target);

    if (target_idx == params.ignore_index || raw_target >= params.num_classes) {
      loss[sample_idx] = 0.0;
    } else {
      let target_logit = logits[row_offset + raw_target];
      let log_sum_exp = log(max(sum_exp, 1e-12)) + max_val;
      loss[sample_idx] = log_sum_exp - target_logit;
    }
  }
}
`;
