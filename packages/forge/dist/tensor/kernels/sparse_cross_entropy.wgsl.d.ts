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
export declare const SPARSE_CROSS_ENTROPY_WGSL = "\nstruct Params {\n  num_samples: u32,\n  num_classes: u32,\n  ignore_index: i32,\n  pad: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: Params;\n@group(0) @binding(1) var<storage, read> logits: array<f32>;\n@group(0) @binding(2) var<storage, read> targets: array<u32>;\n@group(0) @binding(3) var<storage, read_write> loss: array<f32>;\n\nvar<workgroup> s_max: array<f32, 256>;\nvar<workgroup> s_sum: array<f32, 256>;\n\n@compute @workgroup_size(256, 1, 1)\nfn main(\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n  @builtin(workgroup_id) workgroup_id: vec3<u32>\n) {\n  let thread_id = local_id.x;\n  let sample_idx = workgroup_id.x + workgroup_id.y * 65535u;\n\n  if (sample_idx >= params.num_samples) {\n    return;\n  }\n\n  let row_offset = sample_idx * params.num_classes;\n\n  // 1. \uCD5C\uB300\uAC12(Max) \uD0D0\uC0C9 (\uC218\uCE58 \uC548\uC815\uC131 \uD655\uBCF4)\n  var local_max: f32 = -3.402823e+38;\n  for (var c: u32 = thread_id; c < params.num_classes; c = c + 256u) {\n    let val = logits[row_offset + c];\n    local_max = max(local_max, val);\n  }\n  s_max[thread_id] = local_max;\n\n  workgroupBarrier();\n\n  for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {\n    if (thread_id < stride) {\n      s_max[thread_id] = max(s_max[thread_id], s_max[thread_id + stride]);\n    }\n    workgroupBarrier();\n  }\n\n  let max_val = s_max[0];\n\n  // 2. Sum of Exponentials \uACC4\uC0B0\n  var local_sum: f32 = 0.0;\n  for (var c: u32 = thread_id; c < params.num_classes; c = c + 256u) {\n    let val = logits[row_offset + c];\n    local_sum = local_sum + exp(val - max_val);\n  }\n  s_sum[thread_id] = local_sum;\n\n  workgroupBarrier();\n\n  for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {\n    if (thread_id < stride) {\n      s_sum[thread_id] = s_sum[thread_id] + s_sum[thread_id + stride];\n    }\n    workgroupBarrier();\n  }\n\n  let sum_exp = s_sum[0];\n\n  // 3. Thread 0\uC774 NLL Loss \uACC4\uC0B0 \uBC0F \uCD9C\uB825 \uBC84\uD37C\uC5D0 \uAE30\uB85D\n  if (thread_id == 0u) {\n    let raw_target = targets[sample_idx];\n    let target_idx = i32(raw_target);\n\n    if (target_idx == params.ignore_index || raw_target >= params.num_classes) {\n      loss[sample_idx] = 0.0;\n    } else {\n      let target_logit = logits[row_offset + raw_target];\n      let log_sum_exp = log(max(sum_exp, 1e-12)) + max_val;\n      loss[sample_idx] = log_sum_exp - target_logit;\n    }\n  }\n}\n";
