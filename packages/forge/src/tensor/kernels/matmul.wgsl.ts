/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const MATMUL_WGSL = `
/**
 * 이 구조체(Params)는 행렬 곱셈 연산(A x B = C)에 필요한 차원 정보와 오프셋을 제공하기 위해 존재합니다.
 * A행렬은 (M x K), B행렬은 (K x N), C행렬은 (M x N) 차원을 가집니다.
 */
struct Params {
  M: u32, // 행렬 A와 C의 행(Row) 개수입니다.
  N: u32, // 행렬 B와 C의 열(Column) 개수입니다.
  K: u32, // 행렬 A의 열(Column) 개수이자 행렬 B의 행(Row) 개수입니다 (내적을 수행할 길이).
  offsetY: u32, // 워크그룹 파견 한계(dispatch limit)를 우회하기 위해 y축 시작 오프셋을 지정합니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 행렬의 형태 및 오프셋 정보를 담은 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> a: array<f32>; // (M x K) 크기의 첫 번째 입력 행렬 데이터입니다.
@group(0) @binding(2) var<storage, read> b: array<f32>; // (K x N) 크기의 두 번째 입력 행렬 데이터입니다.
@group(0) @binding(3) var<storage, read_write> c: array<f32>; // 결과값(M x N)이 기록될 출력 행렬 데이터입니다.

/**
 * main 함수는 두 행렬 A와 B를 곱하여 결과 행렬 C를 계산합니다.
 * 딥러닝에서 가장 핵심적인 연산인 GEMM(General Matrix Multiply)을 GPU로 분산하여 병렬 처리하기 위해 존재합니다.
 */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // global_id.z 를 X축 타일 오프셋으로 사용
  // dispatcher가 z = ceil(N / (65535*8))만큼 dispatch
  
  // 현재 스레드가 계산을 담당할 출력 행렬 C의 열(Column) 인덱스를 계산합니다.
  // z축 워크그룹 인덱스를 사용하여 1D 한계를 넘는 큰 행렬에 대한 스팬(span)을 지원합니다.
  let col = global_id.x + global_id.z * 65535u * 8u;
  // 현재 스레드가 계산을 담당할 출력 행렬 C의 행(Row) 인덱스를 계산합니다 (오프셋 포함).
  let row = global_id.y + params.offsetY;

  // 계산된 인덱스가 행렬 C의 범위를 초과하는 스레드는 작업을 수행하지 않고 바로 종료합니다.
  if (row >= params.M || col >= params.N) {
    return;
  }

  // A행렬의 row번째 행과 B행렬의 col번째 열 사이의 내적(Dot product)을 누적할 변수입니다.
  var sum: f32 = 0.0;
  
  // 내적을 수행하기 위해 공통 차원인 K번 만큼 루프를 돕니다.
  // A의 원소와 B의 원소를 순차적으로 곱하여 합산합니다.
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  // 계산된 최종 내적 값을 결과 행렬 C의 해당하는 1D 인덱스 위치에 저장합니다.
  c[row * params.N + col] = sum;
}
`;
