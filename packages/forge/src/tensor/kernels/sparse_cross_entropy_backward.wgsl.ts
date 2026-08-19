/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/src/tensor/kernels/sparse_cross_entropy_backward.wgsl.ts
 * Type: WebGPU WGSL Compute Kernel (Fused Sparse Cross-Entropy Backward Gradient)
 * Created: 2026-08-19T01:00:00+09:00
 * ============================================================================
 * WHAT:
 *   Sparse Cross-Entropy의 기울기(grad_logits, [N, C])를 One-Hot 행렬 없이
 *   GPU 상에서 단일 패스 Softmax - Indicator 수식으로 직접 계산하는 역전파 커널입니다.
 * WHY:
 *   O(N * C) 중간 미분 텐서 할당을 완전히 제거하여 거대 어휘집(C=32k~128k) 환경에서
 *   VRAM 메모리 대역폭을 절감하고 초고속 역전파를 지원하기 위함입니다.
 * HOW:
 *   1개 워크그룹(256 스레드)이 1개 배치 샘플을 전담하여, Logits의 Softmax 확률을 구한 후
 *   (prob[c] - (c == target ? 1.0 : 0.0)) * grad_out * reduction_scale을 직접 기록합니다.
 */

export const SPARSE_CROSS_ENTROPY_BACKWARD_WGSL = /* wgsl */ `
struct Params {
  num_samples: u32,
  num_classes: u32,
  ignore_index: i32,
  reduction_scale: f32,
  workgroupsX: u32,
  grad_output_is_scalar: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_logits: array<f32>;

var<workgroup> s_max: array<f32, 256>;
var<workgroup> s_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let sample_idx = workgroup_id.x + workgroup_id.y * params.workgroupsX;

  if (sample_idx >= params.num_samples) {
    return;
  }

  if (params.num_classes == 0u) {
    return;
  }

  let row_offset = sample_idx * params.num_classes;

  // 1. 최대값(Max) 탐색
  var local_max: f32 = -3.402823e+38;
  for (var c: u32 = thread_id; c < params.num_classes; c = c + 256u) {
    let val = logits[row_offset + c];
    if (val == val) {
      local_max = max(local_max, val);
    }
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

  let sum_exp = max(s_sum[0], 1e-12);
  let target_float = targets[sample_idx];
  let rounded = round(target_float);
  let is_target_valid = (target_float == target_float) && (rounded >= 0.0) && (rounded < f32(params.num_classes)) && (i32(rounded) != params.ignore_index);
  let raw_target = select(0u, u32(max(0.0, rounded)), is_target_valid);

  // 스칼라 Loss 역전파 시 0번 인덱스, 샘플별 가중치/벡터 역전파 시 sample_idx를 먼저 안전하게 선택하여 OOB 로드 차단
  let grad_idx = select(sample_idx, 0u, params.grad_output_is_scalar == 1u);
  let g_out = grad_output[grad_idx];
  let scale = g_out * params.reduction_scale;

  // 3. 각 클래스별 기울기 계산: (prob - indicator) * scale
  for (var c: u32 = thread_id; c < params.num_classes; c = c + 256u) {
    let val = logits[row_offset + c];
    let prob = exp(val - max_val) / sum_exp;

    if (!is_target_valid) {
      grad_logits[row_offset + c] = 0.0;
    } else {
      let indicator = select(0.0, 1.0, c == raw_target);
      grad_logits[row_offset + c] = (prob - indicator) * scale;
    }
  }
}
`;
