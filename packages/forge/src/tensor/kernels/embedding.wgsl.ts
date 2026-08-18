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

export const EMBEDDING_WGSL = /* wgsl */ `
struct EmbeddingParams {
  num_tokens: u32,
  embedding_dim: u32,
  vocab_size: u32,
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: EmbeddingParams;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let flat_token_idx = workgroup_id.x + workgroup_id.y * 65535u;

  if (flat_token_idx >= params.num_tokens) {
    return;
  }

  // Float32Array 형태로 전달된 정수 인덱스 읽기 (안전한 정수 캐스팅)
  let raw_idx = index[flat_token_idx];
  var token_id: u32 = 0u;

  if (raw_idx >= 0.0 && raw_idx < f32(params.vocab_size)) {
    token_id = u32(raw_idx);
  } else {
    // Vocab 범위를 벗어난 OOB 인덱스는 0으로 클램프하거나 0 벡터 출력
    token_id = 0u;
  }

  let weight_row_offset = token_id * params.embedding_dim;
  let out_token_offset = flat_token_idx * params.embedding_dim;

  // 64개 스레드가 embedding_dim 차원을 협력 복사
  for (var d: u32 = thread_id; d < params.embedding_dim; d = d + 64u) {
    out[out_token_offset + d] = weight[weight_row_offset + d];
  }
}
`;
