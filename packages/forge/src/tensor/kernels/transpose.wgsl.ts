/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:59:35: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization) (67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const TRANSPOSE_WGSL = `
// 구조체: Params
// 목적: 행렬 전치(Transpose) 연산에 필요한 차원 정보를 전달합니다.
// 작동 방식: 행(M), 열(N), 배치(B) 크기를 받아 다차원 배열의 인덱스를 계산할 수 있게 합니다.
struct Params {
  // 변수: M
  // 목적: 변환 전 원본 행렬의 행(Row) 개수입니다.
  // 작동 방식: 전치 후에는 이 값이 열의 개수가 됩니다.
  M: u32,
  // 변수: N
  // 목적: 변환 전 원본 행렬의 열(Column) 개수입니다.
  // 작동 방식: 전치 후에는 이 값이 행의 개수가 됩니다.
  N: u32,
  // 변수: B
  // 목적: 배치(Batch) 크기를 의미합니다.
  // 작동 방식: 여러 개의 독립적인 행렬(배치)을 동시에 전치할 수 있게 합니다.
  B: u32,
};

// 변수: params
// 목적: 셰이더 실행 시 필요한 차원(M, N, B) 정보를 담은 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0에 매핑되어 인덱스 계산의 기준값으로 사용됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 전치하기 전의 원본 다차원 배열(배치 포함 3차원 구조) 데이터입니다.
// 작동 방식: 바인딩 1에 읽기 전용 스토리지 버퍼로 바인딩됩니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: out
// 목적: 행과 열이 뒤바뀐 전치 결과를 저장할 출력 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되어 계산된 데이터가 저장됩니다.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: 배치 차원을 유지한 채로 행(Row)과 열(Column)의 위치를 바꾸는 전치 연산을 수행합니다.
// 작동 방식: 3차원 글로벌 인덱스(x, y, z)를 각각 (row, col, batch)로 매핑하고 변환 공식을 적용합니다.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: row
  // 목적: 원본 행렬 기준에서의 행 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 x 성분(global_id.x)을 사용합니다.
  let row = global_id.x;

  // 변수: col
  // 목적: 원본 행렬 기준에서의 열 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 y 성분(global_id.y)을 사용합니다.
  let col = global_id.y;

  // 변수: batch
  // 목적: 현재 처리 중인 배치의 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 z 성분(global_id.z)을 사용합니다.
  let batch = global_id.z;
  
  // 제어문: if
  // 목적: 패딩이나 워크그룹 크기 맞춤으로 인해 실제 데이터 범위를 초과한 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: row, col, batch가 각각 M, N, B보다 작은지 확인합니다.
  if (row < params.M && col < params.N && batch < params.B) {
    // 변수: in_idx
    // 목적: 1차원으로 평면화된 원본 배열에서 읽어올 요소의 인덱스를 계산합니다.
    // 작동 방식: '배치 오프셋 + 행 오프셋 + 열' (batch * M * N + row * N + col) 공식을 사용합니다.
    let in_idx = batch * (params.M * params.N) + row * params.N + col;

    // 변수: out_idx
    // 목적: 전치된 결과를 저장할 출력 배열의 1차원 평면화 인덱스를 계산합니다.
    // 작동 방식: 행과 열의 기준 크기가 바뀌므로 '배치 오프셋 + 새로운 행 오프셋 + 새로운 열' (batch * M * N + col * M + row)로 계산합니다.
    let out_idx = batch * (params.M * params.N) + col * params.M + row;

    // 연산: out[out_idx] 할당
    // 목적: 계산된 위치에 원본 데이터를 복사하여 전치를 완료합니다.
    // 작동 방식: 원본 위치(in_idx)의 값을 읽어 목표 위치(out_idx)에 기록합니다.
    out[out_idx] = input[in_idx];
  }
}
`;
