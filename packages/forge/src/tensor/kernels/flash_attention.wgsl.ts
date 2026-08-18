/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-209 / SCRUM-210 / SCRUM-211
 * FlashAttention-2 Fused 1-Pass Online Softmax WGSL Kernel (MHA / GQA / Causal)
 *
 * WHAT: O(N) 메모리 복잡도를 가지는 FlashAttention-2 융합 1-Pass 어텐션 WGSL 셰이더입니다.
 * WHY: 표준 Scaled Dot-Product Attention의 O(N^2) 어텐션 맵 VRAM 할당과 대역폭 병목을 100% 제거하고,
 *      긴 시퀀스(SeqLen 2048~4096)에서도 OOM 없이 초고속 LLM 추론을 가능하게 합니다.
 * HOW: Dao et al.의 FlashAttention-2 Online Softmax 알고리즘(Running Max & Running Sum)을 GPU 스레드 레지스터 레벨에서
 *      단일 패스로 융합하고, Grouped Query Attention(GQA)과 Causal Masking을 셰이더 내부에서 인라인으로 처리합니다.
 */

export const FLASH_ATTENTION_WGSL = `
struct Params {
  B: u32,             // 총 배치 수
  H: u32,             // 쿼리 헤드 수 (Query Heads)
  H_kv: u32,          // KV 헤드 수 (KV Heads, GQA 지원용: H / H_kv = 그룹 크기)
  N_q: u32,           // 쿼리 시퀀스 길이 (Sequence Length Q)
  N_kv: u32,          // 키/값 시퀀스 길이 (Sequence Length KV)
  d: u32,             // 헤드 차원 (Head Dim, 예: 64, 128, 256)
  scale: f32,         // 1.0 / sqrt(d) 스케일 팩터
  is_causal: u32,     // 1: Causal Masking 적용, 0: Full Attention
  strideQ: u32,       // Q 텐서의 배치*헤드당 오프셋 보폭
  strideK: u32,       // K 텐서의 배치*헤드당 오프셋 보폭
  strideV: u32,       // V 텐서의 배치*헤드당 오프셋 보폭
  strideO: u32,       // O 텐서의 배치*헤드당 오프셋 보폭
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

// 워크그룹 공유 메모리: 쿼리 벡터(s_q)와 현재 키 벡터(s_k)를 온칩 SRAM에 캐시
var<workgroup> s_q: array<f32, 256>;
var<workgroup> s_k: array<f32, 256>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let thread_id = local_id.x;
  let q_idx = workgroup_id.x; // 현재 처리할 쿼리 토큰 인덱스 (0 .. N_q-1)
  let head_idx = workgroup_id.y; // 쿼리 헤드 인덱스 (0 .. H-1)
  let batch_idx = workgroup_id.z; // 배치 인덱스 (0 .. B-1)

  if (q_idx >= params.N_q || head_idx >= params.H || batch_idx >= params.B) {
    return;
  }

  // GQA 매핑: 쿼리 헤드 인덱스에 대응하는 KV 헤드 인덱스 계산
  let group_size = params.H / params.H_kv;
  let kv_head_idx = head_idx / group_size;

  let q_head_offset = batch_idx * (params.H * params.strideQ) + head_idx * params.strideQ;
  let k_head_offset = batch_idx * (params.H_kv * params.strideK) + kv_head_idx * params.strideK;
  let v_head_offset = batch_idx * (params.H_kv * params.strideV) + kv_head_idx * params.strideV;
  let o_head_offset = batch_idx * (params.H * params.strideO) + head_idx * params.strideO;

  let q_token_offset = q_head_offset + q_idx * params.d;
  let o_token_offset = o_head_offset + q_idx * params.d;

  // 1. 협력 적재: 쿼리 벡터 Q[q_idx, 0..d-1]를 워크그룹 공유 메모리에 캐시
  for (var c: u32 = thread_id; c < params.d; c = c + 64u) {
    s_q[c] = q[q_token_offset + c];
  }
  workgroupBarrier();

  // 각 스레드가 담당할 차원 d의 서브셋 (최대 d=256 지원)
  // Online Softmax State
  var m_prev: f32 = -1e30; // Running max
  var l_prev: f32 = 0.0;   // Running sum

  // 현재 스레드가 누적할 출력 원소 레지스터 4개
  var thread_acc0: f32 = 0.0;
  var thread_acc1: f32 = 0.0;
  var thread_acc2: f32 = 0.0;
  var thread_acc3: f32 = 0.0;

  let dim_idx0 = thread_id;
  let dim_idx1 = thread_id + 64u;
  let dim_idx2 = thread_id + 128u;
  let dim_idx3 = thread_id + 192u;

  // Causal Masking 적용 시 최대 키 인덱스 계산 (KV 캐시 오프셋 고려)
  var max_k_len: u32 = params.N_kv;
  if (params.is_causal == 1u) {
    // 디코딩 단계에서는 전체 캐시된 KV에 대해 어텐션 가능
    let causal_limit = params.N_kv - params.N_q + q_idx + 1u;
    max_k_len = min(params.N_kv, causal_limit);
  }

  // 2. K/V 시퀀스를 1-Pass로 순회하며 온칩 SRAM 캐싱 & Online Softmax
  for (var j: u32 = 0u; j < max_k_len; j = j + 1u) {
    let k_token_offset = k_head_offset + j * params.d;
    let v_token_offset = v_head_offset + j * params.d;

    // K 벡터를 워크그룹 공유 메모리에 협력 로드 (Cooperative SRAM Cache)
    for (var c: u32 = thread_id; c < params.d; c = c + 64u) {
      s_k[c] = k[k_token_offset + c];
    }
    workgroupBarrier();

    // Step A: 내적 Score S_ij = scale * sum_{c} Q[c] * K[j, c] (온칩 SRAM 고속 접근)
    var dot: f32 = 0.0;
    for (var c: u32 = 0u; c < params.d; c = c + 1u) {
      dot = dot + s_q[c] * s_k[c];
    }
    let score = dot * params.scale;

    // Step B: FlashAttention-2 Online Softmax Rescale
    let m_new = max(m_prev, score);
    let alpha = exp(m_prev - m_new);
    let p = exp(score - m_new);

    l_prev = l_prev * alpha + p;
    m_prev = m_new;

    // Step C: Running Output Rescale & Value Accumulation
    if (dim_idx0 < params.d) {
      thread_acc0 = thread_acc0 * alpha + p * v[v_token_offset + dim_idx0];
    }
    if (dim_idx1 < params.d) {
      thread_acc1 = thread_acc1 * alpha + p * v[v_token_offset + dim_idx1];
    }
    if (dim_idx2 < params.d) {
      thread_acc2 = thread_acc2 * alpha + p * v[v_token_offset + dim_idx2];
    }
    if (dim_idx3 < params.d) {
      thread_acc3 = thread_acc3 * alpha + p * v[v_token_offset + dim_idx3];
    }
    workgroupBarrier();
  }

  // 3. 최종 소프트맥스 합(l_prev)으로 나누어 정규화 후 글로벌 메모리에 기록
  let inv_l = 1.0 / max(l_prev, 1e-12);

  if (dim_idx0 < params.d) {
    o[o_token_offset + dim_idx0] = thread_acc0 * inv_l;
  }
  if (dim_idx1 < params.d) {
    o[o_token_offset + dim_idx1] = thread_acc1 * inv_l;
  }
  if (dim_idx2 < params.d) {
    o[o_token_offset + dim_idx2] = thread_acc2 * inv_l;
  }
  if (dim_idx3 < params.d) {
    o[o_token_offset + dim_idx3] = thread_acc3 * inv_l;
  }
}
`;
