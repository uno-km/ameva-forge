/**
 * 파일 생성일: 2026-08-12
 * 수정일: 2026-08-18 (Release 2.0 SCRUM-204 16x16 Shared Memory Tiled Batched MatMul)
 *
 * WHAT: 16x16 워크그룹 공유 메모리(Shared Memory) 기반 4D Batched General Matrix Multiply 커널입니다.
 * WHY: Multi-Head Attention (MHA/GQA)에서 QK^T 및 Attn*V 연산의 글로벌 메모리 병목을 제거하기 위해 존재합니다.
 * HOW: 각 배치 인덱스(global_id.z) 내에서 16x16 A/B 타일을 온칩 SRAM에 적재하고 workgroupBarrier()로 동기화하여 고속 배치 GEMM을 수행합니다.
 */

export const BATCHED_MATMUL_WGSL = `
struct Params {
  B: u32,       // 총 배치 수 (예: Batch * NumHeads)
  M: u32,       // 행렬 A/C의 행 개수
  N: u32,       // 행렬 B/C의 열 개수
  K: u32,       // 공통 내적 차원
  strideA: u32, // 배치당 A 오프셋 보폭
  strideB: u32, // 배치당 B 오프셋 보폭
  strideC: u32, // 배치당 C 오프셋 보폭
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let local_row = local_id.y;
  let local_col = local_id.x;

  let global_row_c = workgroup_id.y * 16u + local_row;
  let global_col_c = workgroup_id.x * 16u + local_col;
  let batch = workgroup_id.z;

  if (batch >= params.B) {
    return;
  }

  let batch_a_offset = batch * params.strideA;
  let batch_b_offset = batch * params.strideB;
  let batch_c_offset = batch * params.strideC;

  let num_tiles = (params.K + 15u) / 16u;
  var acc: f32 = 0.0;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let global_row_a = global_row_c;
    let global_col_a = t * 16u + local_col;

    if (global_row_a < params.M && global_col_a < params.K) {
      tileA[local_row][local_col] = a[batch_a_offset + global_row_a * params.K + global_col_a];
    } else {
      tileA[local_row][local_col] = 0.0;
    }

    let global_row_b = t * 16u + local_row;
    let global_col_b = global_col_c;

    if (global_row_b < params.K && global_col_b < params.N) {
      tileB[local_row][local_col] = b[batch_b_offset + global_row_b * params.N + global_col_b];
    } else {
      tileB[local_row][local_col] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      acc = acc + tileA[local_row][k] * tileB[k][local_col];
    }

    workgroupBarrier();
  }

  if (global_row_c < params.M && global_col_c < params.N) {
    c[batch_c_offset + global_row_c * params.N + global_col_c] = acc;
  }
}
`;
