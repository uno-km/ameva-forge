/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/src/tensor/kernels/embedding_backward.wgsl.ts
 * Type: WebGPU WGSL Compute Kernel (Native Embedding Backward Gradient Accumulation)
 * Created: 2026-08-18T23:36:00+09:00
 * ============================================================================
 * WHAT:
 *   임베딩 순전파의 출력 기울기(grad_output, [B, L, D])와 토큰 인덱스(index, [B, L])를 입력받아
 *   임베딩 가중치 행렬의 기울기(grad_weight, [Vocab, D])를 계산하는 WebGPU Native 역전파 커널입니다.
 * WHY:
 *   atomicAdd 없이도 100% 표준 WebGPU WGSL 환경에서 임베딩 계층의 역전파를
 *   완전 Lock-free 병렬 누산으로 안전하게 수행하기 위함입니다.
 * HOW:
 *   출력 grad_weight[v, d]의 각 성분을 독립적인 GPU 스레드에 매핑하고,
 *   토큰 인덱스 버퍼를 스캔하여 index[t] == v인 경우 grad_output[t, d]를 결정론적으로 합산합니다.
 */
export declare const EMBEDDING_BACKWARD_WGSL = "\nstruct EmbeddingBackwardParams {\n  num_tokens: u32,\n  embedding_dim: u32,\n  vocab_size: u32,\n  total_weight_elements: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: EmbeddingBackwardParams;\n@group(0) @binding(1) var<storage, read> grad_output: array<f32>;\n@group(0) @binding(2) var<storage, read> index: array<u32>;\n@group(0) @binding(3) var<storage, read_write> grad_weight: array<f32>;\n\n@compute @workgroup_size(64, 1, 1)\nfn main(\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n  @builtin(workgroup_id) workgroup_id: vec3<u32>\n) {\n  let thread_id = local_id.x;\n  let flat_idx = (workgroup_id.x + workgroup_id.y * 65535u) * 64u + thread_id;\n\n  if (flat_idx >= params.total_weight_elements) {\n    return;\n  }\n\n  let vocab_id = flat_idx / params.embedding_dim;\n  let d = flat_idx % params.embedding_dim;\n\n  var acc: f32 = 0.0;\n\n  // index[t] == vocab_id\uC778 \uBAA8\uB4E0 \uD1A0\uD070 t\uC5D0 \uB300\uD574 grad_output[t, d] \uB204\uC0B0 (Lock-Free & Deterministic)\n  for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {\n    if (index[t] == vocab_id) {\n      let grad_out_offset = t * params.embedding_dim + d;\n      acc = acc + grad_output[grad_out_offset];\n    }\n  }\n\n  grad_weight[flat_idx] = acc;\n}\n";
