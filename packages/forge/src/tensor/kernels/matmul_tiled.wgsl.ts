/**
 * 파일 생성일: 2026-08-18 20:12:00 +0900
 * AMEVA-Forge Release 2.0: SCRUM-201 / SCRUM-207
 * Tiled General Matrix Multiply (GEMM) using Workgroup Shared Memory (16x16 Tile)
 *
 * WHAT: 16x16 워크그룹 공유 메모리(Shared Memory)를 활용한 타일드 행렬곱(Tiled MatMul) WGSL 셰이더입니다.
 * WHY: Naive MatMul의 극심한 글로벌 메모리 대역폭 병목을 해결하고, 연산 처리율을 3.5x~5x 향상시키기 위해 존재합니다.
 * HOW: 각 워크그룹(256 스레드)이 16x16 크기의 A, B 타일을 공유 메모리에 협력하여 로드(Cooperative Load)한 후,
 *      workgroupBarrier() 동기화를 거쳐 캐시된 타일 내적을 계산하고, M/N/K 비정렬 경계값(Non-multiples of 16)을
 *      Zero-Padding과 Bounds Guard로 안전하게 처리합니다.
 */

export const MATMUL_TILED_WGSL = `
struct Params {
  M: u32,       // 행렬 A와 C의 행(Row) 개수
  N: u32,       // 행렬 B와 C의 열(Column) 개수
  K: u32,       // 행렬 A의 열이자 행렬 B의 행 개수 (내적 축 길이)
  offsetY: u32, // 2D 디스패치 파티셔닝 오프셋
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

// 16x16 워크그룹 공유 메모리 타일 선언 (각 1024 바이트, 총 2048 바이트 할당)
var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let local_row = local_id.y;
  let local_col = local_id.x;

  // 출력 행렬 C에서 현재 스레드가 담당할 글로벌 2D 좌표 계산 (Z축 오버플로우 스팬 포함)
  let global_row_c = workgroup_id.y * 16u + local_row + params.offsetY;
  let global_col_c = (workgroup_id.x + workgroup_id.z * 65535u) * 16u + local_col;

  // K차원을 16 크기의 타일로 분할한 총 타일 개수 (올림 처리)
  let num_tiles = (params.K + 15u) / 16u;

  var acc: f32 = 0.0;

  // K차원을 따라 타일 단위로 순차 이동하며 내적 누적
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // 1. 행렬 A 타일 협력 적재 (Cooperative Tile Load) with Zero-Padding Boundary Guard
    let global_row_a = global_row_c;
    let global_col_a = t * 16u + local_col;

    if (global_row_a < params.M && global_col_a < params.K) {
      tileA[local_row][local_col] = a[global_row_a * params.K + global_col_a];
    } else {
      tileA[local_row][local_col] = 0.0; // SCRUM-207: 비정렬 경계 제로 패딩
    }

    // 2. 행렬 B 타일 협력 적재 (Cooperative Tile Load) with Zero-Padding Boundary Guard
    let global_row_b = t * 16u + local_row;
    let global_col_b = global_col_c;

    if (global_row_b < params.K && global_col_b < params.N) {
      tileB[local_row][local_col] = b[global_row_b * params.N + global_col_b];
    } else {
      tileB[local_row][local_col] = 0.0; // SCRUM-207: 비정렬 경계 제로 패딩
    }

    // 모든 워크그룹 스레드가 공유 메모리에 타일 로드를 완료할 때까지 대기
    workgroupBarrier();

    // 3. 공유 메모리에 적재된 16개 원소에 대해 빠른 내적 수행
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      acc = acc + tileA[local_row][k] * tileB[k][local_col];
    }

    // 다음 타일을 로드하기 전에 모든 스레드가 현재 공유 메모리 읽기를 마칠 때까지 대기
    workgroupBarrier();
  }

  // 4. 유효한 행렬 C 경계 내의 스레드만 글로벌 메모리에 최종 결과 기록
  if (global_row_c < params.M && global_col_c < params.N) {
    c[global_row_c * params.N + global_col_c] = acc;
  }
}
`;
