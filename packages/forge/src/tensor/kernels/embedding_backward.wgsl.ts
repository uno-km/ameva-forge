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
  workgroupsX: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
};

@group(0) @binding(0) var<uniform> params: EmbeddingBackwardParams;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_weight: array<f32>;

var<workgroup> s_match_count: atomic<u32>;
var<workgroup> s_matches: array<u32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let base_workgroup_idx = (workgroup_id.x + workgroup_id.y * params.workgroupsX) * 64u;
  let flat_idx = base_workgroup_idx + thread_id;

  let start_vocab = base_workgroup_idx / params.embedding_dim;
  let end_vocab = (base_workgroup_idx + 63u) / params.embedding_dim;

  // 단일 워크그룹 내 모든 스레드가 동일한 vocab_id를 처리하는 경우 (표준 LLM: embedding_dim >= 64):
  // 64개 단위로 청크 순회(Chunked Cooperative Scan)를 수행하여 64개를 초과하는 임의의 출현 횟수도 절단 없이 100% 완전 누적
  if (start_vocab == end_vocab && start_vocab < params.vocab_size) {
    let target_v = start_vocab;
    let d = flat_idx % params.embedding_dim;
    var acc: f32 = 0.0;

    let num_chunks = (params.num_tokens + 63u) / 64u;
    for (var chunk: u32 = 0u; chunk < num_chunks; chunk = chunk + 1u) {
      if (thread_id == 0u) {
        atomicStore(&s_match_count, 0u);
      }
      workgroupBarrier();

      let t = chunk * 64u + thread_id;
      if (t < params.num_tokens) {
        let raw_val = index[t];
        let rounded = round(raw_val);
        if (raw_val == raw_val && rounded >= 0.0 && rounded < f32(params.vocab_size)) {
          let raw_token_id = u32(rounded);
          if (raw_token_id == target_v) {
            let slot = atomicAdd(&s_match_count, 1u);
            if (slot < 64u) {
              s_matches[slot] = t;
            }
          }
        }
      }
      workgroupBarrier();

      let count = min(atomicLoad(&s_match_count), 64u);
      if (count > 0u && flat_idx < params.total_weight_elements) {
        for (var m: u32 = 0u; m < count; m = m + 1u) {
          let matched_t = s_matches[m];
          let grad_out_offset = matched_t * params.embedding_dim + d;
          acc = acc + grad_output[grad_out_offset];
        }
      }
      workgroupBarrier();
    }

    if (flat_idx < params.total_weight_elements) {
      grad_weight[flat_idx] = acc;
    }
    return;
  }

  // 워크그룹이 경계를 넘거나 소형 임베딩 차원인 경우 일반 스캔
  if (flat_idx >= params.total_weight_elements) {
    return;
  }

  let vocab_id = flat_idx / params.embedding_dim;
  let d = flat_idx % params.embedding_dim;

  var acc: f32 = 0.0;
  for (var t: u32 = 0u; t < params.num_tokens; t = t + 1u) {
    let raw_val = index[t];
    let rounded = round(raw_val);
    if (raw_val == raw_val && rounded >= 0.0 && rounded < f32(params.vocab_size)) {
      let raw_token_id = u32(rounded);
      if (raw_token_id == vocab_id) {
        let grad_out_offset = t * params.embedding_dim + d;
        acc = acc + grad_output[grad_out_offset];
      }
    }
  }

  grad_weight[flat_idx] = acc;
}
`;
