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

export const EMBEDDING_BACKWARD_WGSL = /* wgsl */ `
struct EmbeddingBackwardParams {
  num_tokens: u32,
  embedding_dim: u32,
  vocab_size: u32,
  total_weight_elements: u32,
};

@group(0) @binding(0) var<uniform> params: EmbeddingBackwardParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<u32>;
@group(0) @binding(3) var<storage, read_write> grad_weight: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let flat_idx = (workgroup_id.x + workgroup_id.y * 65535u) * 64u + thread_id;

  if (flat_idx >= params.total_weight_elements) {
    return;
  }

  let vocab_id = flat_idx / params.embedding_dim;
  let d = flat_idx % params.embedding_dim;

  var acc: f32 = 0.0;

  // index[t] == vocab_id인 모든 토큰 t에 대해 grad_output[t, d] 누산 (Lock-Free & Deterministic)
  for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {
    if (index[t] == vocab_id) {
      let grad_out_offset = t * params.embedding_dim + d;
      acc = acc + grad_output[grad_out_offset];
    }
  }

  grad_weight[flat_idx] = acc;
}
`;
