/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const SUM_AXIS_WGSL = `
// 구조체: Params
// 목적: 특정 축(Axis)을 기준으로 합계를 구할 때 필요한 차원 정보를 제공합니다.
// 작동 방식: 행(M)과 열(N)의 크기를 정의하고, 16바이트 메모리 정렬을 맞춥니다.
struct Params {
  // 변수: M
  // 목적: 누적(sum) 연산이 수행될 행(Row)의 크기(축의 길이)를 지정합니다.
  // 작동 방식: 반복문에서 행의 인덱스가 M에 도달할 때까지 합계를 구합니다.
  M: u32,
  // 변수: N
  // 목적: 스레드들이 병렬로 처리할 열(Column)의 개수입니다.
  // 작동 방식: 각 스레드가 N개의 열 중 하나를 담당하여 각 열의 합계를 독립적으로 계산합니다.
  N: u32,
  // 변수: pad1
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 읽기 성능 저하를 방지하기 위해 빈 공간을 둡니다.
  pad1: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 읽기 성능 저하를 방지하기 위해 빈 공간을 둡니다.
  pad2: u32,
};

// 변수: params
// 목적: 커널 실행 시 M, N 등의 차원 정보를 담아 전달하는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0에 매핑되어 워크그룹 내에서 공유됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 합계를 구할 2차원(혹은 1차원으로 평면화된) 배열 데이터를 저장하는 버퍼입니다.
// 작동 방식: 바인딩 1에 할당되며 읽기 전용으로 접근합니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 목적: 특정 축을 기준으로 축소(Reduce)된 결과를 저장할 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되며, N 크기의 배열로 결과가 쓰여집니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: 주어진 데이터의 첫 번째 축(Row)을 기준으로 열 단위 합계를 병렬 연산합니다.
// 작동 방식: 각 스레드가 하나의 열(col)을 담당하고 모든 행(row)을 순회하며 합산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: col
  // 목적: 현재 스레드가 담당할 열(Column)의 인덱스입니다.
  // 작동 방식: global_id.x 값을 이용하여 계산할 열 위치를 특정합니다.
  let col = global_id.x;
  
  // 제어문: if
  // 목적: 할당된 스레드의 인덱스가 실제 열 개수(N)를 초과하는지 검사합니다.
  // 작동 방식: col이 N 이상이면 유효하지 않은 스레드이므로 즉시 종료(return)합니다.
  if (col >= params.N) {
    return;
  }
  
  // 변수: sum
  // 목적: 특정 열에 대한 총합을 누적할 로컬 변수입니다.
  // 작동 방식: 0.0으로 초기화된 후 루프를 돌면서 요소의 값을 계속 더해나갑니다.
  var sum = 0.0;
  
  // 반복문: for
  // 목적: 행(Row) 방향으로 데이터를 탐색하며 각 요소를 더합니다.
  // 작동 방식: row를 0부터 M-1까지 1씩 증가시키며 \`sum += input[row * N + col]\` 연산을 수행합니다.
  for (var row = 0u; row < params.M; row = row + 1u) {
    sum += input[row * params.N + col];
  }
  
  // 연산: output 배열 기록
  // 목적: 반복문이 끝난 후 계산된 최종 열 합계를 결과 버퍼에 저장합니다.
  // 작동 방식: 해당 스레드가 담당한 열 인덱스(col) 위치에 누적된 sum을 기록합니다.
  output[col] = sum;
}
`;
