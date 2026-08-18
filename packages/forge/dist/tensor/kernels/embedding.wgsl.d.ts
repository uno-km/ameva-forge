/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/src/tensor/kernels/embedding.wgsl.ts
 * Type: WebGPU WGSL Compute Kernel (Native Embedding Lookup)
 * Created: 2026-08-18T23:18:00+09:00
 * ============================================================================
 * WHAT:
 *   단어/토큰 인덱스 텐서([B, L])를 입력받아 임베딩 가중치 행렬([Vocab, D])에서
 *   해당 행 벡터를 추출하여 [B, L, D] 텐서를 생성하는 WebGPU Native 임베딩 룩업 커널입니다.
 * WHY:
 *   다차원 gather 커널을 오용할 때 발생하는 스키마 불일치 및 인덱스 OOB 읽기 오류를
 *   원천 차단하고, 2D 그리드 디스패치를 통해 수백만 토큰까지 안전하고 빠르게 룩업하기 위함입니다.
 * HOW:
 *   워크그룹당 1개의 토큰 인덱스를 처리하며, 64개 워크그룹 스레드가 협력하여
 *   embedding_dim 차원의 부동소수점 데이터를 고속 복사합니다.
 */
export declare const EMBEDDING_WGSL = "\nstruct EmbeddingParams {\n  num_tokens: u32,\n  embedding_dim: u32,\n  vocab_size: u32,\n  pad: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: EmbeddingParams;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read> index: array<u32>;\n@group(0) @binding(3) var<storage, read_write> out: array<f32>;\n\n@compute @workgroup_size(64, 1, 1)\nfn main(\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n  @builtin(workgroup_id) workgroup_id: vec3<u32>\n) {\n  let thread_id = local_id.x;\n  let flat_token_idx = workgroup_id.x + workgroup_id.y * 65535u;\n\n  if (flat_token_idx >= params.num_tokens) {\n    return;\n  }\n\n  // 32\uBE44\uD2B8 \uC815\uC218 \uD1A0\uD070 ID\uB97C \uC9C1\uC811 u32\uB85C \uC77D\uC5B4 \uBE44\uD2B8 \uC624\uB3C5 \uBC0F Float32 \uCD95\uD1F4\uB97C \uC6D0\uCC9C \uCC28\uB2E8\n  let raw_token_id = index[flat_token_idx];\n  var token_id: u32 = 0u;\n\n  if (raw_token_id < params.vocab_size) {\n    token_id = raw_token_id;\n  } else {\n    // Vocab \uBC94\uC704\uB97C \uBC97\uC5B4\uB09C OOB \uC778\uB371\uC2A4\uB294 0\uC73C\uB85C \uC548\uC804\uD558\uAC8C \uD074\uB7A8\uD504\n    token_id = 0u;\n  }\n\n  let weight_row_offset = token_id * params.embedding_dim;\n  let out_token_offset = flat_token_idx * params.embedding_dim;\n\n  // 64\uAC1C \uC2A4\uB808\uB4DC\uAC00 embedding_dim \uCC28\uC6D0\uC744 \uD611\uB825 \uBCF5\uC0AC\n  for (var d: u32 = thread_id; d < params.embedding_dim; d = d + 64u) {\n    out[out_token_offset + d] = weight[weight_row_offset + d];\n  }\n}\n";
