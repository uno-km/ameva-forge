/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const BATCHED_MATMUL_WGSL = `
/**
 * @struct Params
 * @brief 배치 행렬 곱셈(Batched Matrix Multiplication)을 제어하기 위한 행렬의 차원 크기와 스트라이드(stride) 정보를 저장합니다. (What)
 * 입력 행렬 텐서 A와 B의 형태(M, N, K)와 연속적인 배치 접근을 위한 메모리 오프셋을 계산할 때 사용하기 위해 정의되었습니다. (Why)
 */
struct Params {
  // 배치(Batch)의 개수입니다. 한 번에 여러 쌍의 행렬 곱셈을 병렬 처리하기 위한 차원입니다.
  B: u32,
  // 결과 행렬(C)과 왼쪽 행렬(A)의 행(Row) 개수입니다.
  M: u32,
  // 결과 행렬(C)과 오른쪽 행렬(B)의 열(Column) 개수입니다.
  N: u32,
  // 왼쪽 행렬(A)의 열 개수이자 오른쪽 행렬(B)의 행 개수로, 내적(Dot product)이 이루어지는 공통 차원의 길이입니다.
  K: u32,
  // 왼쪽 행렬(A)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideA: u32,
  // 오른쪽 행렬(B)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideB: u32,
  // 결과 행렬(C)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideC: u32,
};

// params: 배치 크기 및 행렬 차원 정보를 GPU 스레드들에게 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 첫 번째(왼쪽) 입력 행렬 데이터들을 담고 있는 1차원 배열(읽기 전용)입니다.
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 두 번째(오른쪽) 입력 행렬 데이터들을 담고 있는 1차원 배열(읽기 전용)입니다.
@group(0) @binding(2) var<storage, read> b: array<f32>;
// c: 행렬 곱셈의 결과가 저장될 출력 배열(읽기/쓰기 가능)입니다.
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

/**
 * @function main
 * @brief 주어진 배치(Batch)에 대해 행렬 A와 B의 내적을 수행하여 행렬 C의 각 요소를 계산합니다. (What)
 * 어텐션 메커니즘 등 신경망 구조에서 다중 배치의 텐서를 한 번에 곱하기 위해 (Why) 3차원 그리드로 병렬 실행됩니다.
 * 
 * @param global_id 워크그룹과 스레드의 3차원 인덱스입니다. (x: 열(Column), y: 행(Row), z: 배치(Batch)를 나타냅니다.) (How)
 */
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 스레드의 x 인덱스로, 연산할 결과 행렬 C의 열(Column) 위치를 할당합니다.
  let col = global_id.x;
  // 스레드의 y 인덱스로, 연산할 결과 행렬 C의 행(Row) 위치를 할당합니다.
  let row = global_id.y;
  // 스레드의 z 인덱스로, 현재 처리할 배치(Batch) 번호를 할당합니다.
  let batch = global_id.z;

  // 할당된 인덱스들이 지정된 행렬 크기나 배치 수를 초과하는지 검사합니다. (What)
  // 워크그룹 크기(8x8)로 인해 남는 스레드가 유효하지 않은 메모리에 접근하는 것을 방지하기 위함입니다. (Why)
  if (row >= params.M || col >= params.N || batch >= params.B) {
    return;
  }

  // 1차원 배열 A에서 현재 배치의 현재 행이 시작되는 오프셋을 계산합니다. (How)
  let a_offset = batch * params.strideA + row * params.K;
  // 1차원 배열 B에서 현재 배치의 현재 열이 시작되는 오프셋을 계산합니다.
  let b_offset = batch * params.strideB + col;
  // 1차원 결과 배열 C에서 현재 배치의 위치(row, col)에 해당하는 저장 인덱스를 계산합니다.
  let c_offset = batch * params.strideC + row * params.N + col;

  // 내적(Dot product)을 누적하기 위한 실수형 변수를 선언하고 0으로 초기화합니다. (What)
  var sum: f32 = 0.0;
  
  // 공통 차원인 K번만큼 반복하여 행렬 A의 특정 행과 행렬 B의 특정 열의 요소들을 곱하고 더합니다. (How)
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    // 행렬 A에서는 열(k) 방향으로 이동하고, 행렬 B에서는 행(k) 방향으로 이동(B의 행 길이인 N만큼 점프)하면서 값을 곱하여 sum에 누적시킵니다. (How)
    sum = sum + a[a_offset + k] * b[b_offset + k * params.N];
  }

  // 계산된 내적 최종 결과(sum)를 출력 배열 C의 오프셋 위치에 저장합니다. (What)
  c[c_offset] = sum;
}
`;
