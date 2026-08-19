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
export declare const EMBEDDING_WGSL = "\nstruct EmbeddingParams {\n  num_tokens: u32,\n  embedding_dim: u32,\n  vocab_size: u32,\n  workgroupsX: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: EmbeddingParams;\n@group(0) @binding(1) var<storage, read> weight: array<f32>;\n@group(0) @binding(2) var<storage, read> index: array<f32>;\n@group(0) @binding(3) var<storage, read_write> out: array<f32>;\n\n@compute @workgroup_size(64, 1, 1)\nfn main(\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n  @builtin(workgroup_id) workgroup_id: vec3<u32>\n) {\n  let thread_id = local_id.x;\n  let flat_token_idx = workgroup_id.x + workgroup_id.y * params.workgroupsX;\n\n  if (flat_token_idx >= params.num_tokens) {\n    return;\n  }\n\n  let raw_val = index[flat_token_idx];\n  let out_token_offset = flat_token_idx * params.embedding_dim;\n  let rounded = round(raw_val);\n\n  // NaN\uC774\uAC70\uB098 \uBC18\uC62C\uB9BC \uACB0\uACFC\uAC00 \uC74C\uC218\uC774\uAC70\uB098 \uC5B4\uD718\uC9D1 \uD06C\uAE30 \uC774\uC0C1\uC778 \uC778\uB371\uC2A4\uB294 0\uBC88 \uD1A0\uD070\uC73C\uB85C \uC624\uC5FC\uC2DC\uD0A4\uC9C0 \uC54A\uACE0 0.0 \uBCA1\uD130 \uAE30\uB85D\n  if (raw_val != raw_val || rounded < 0.0 || rounded >= f32(params.vocab_size)) {\n    for (var d: u32 = thread_id; d < params.embedding_dim; d = d + 64u) {\n      out[out_token_offset + d] = 0.0;\n    }\n  } else {\n    let token_id = u32(rounded);\n    let weight_row_offset = token_id * params.embedding_dim;\n    // 64\uAC1C \uC2A4\uB808\uB4DC\uAC00 embedding_dim \uCC28\uC6D0\uC744 \uD611\uB825 \uBCF5\uC0AC\n    for (var d: u32 = thread_id; d < params.embedding_dim; d = d + 64u) {\n      out[out_token_offset + d] = weight[weight_row_offset + d];\n    }\n  }\n}\n";
