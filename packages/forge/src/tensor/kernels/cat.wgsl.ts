/**
 * 생성일 (Created): 2026-08-12 12:23:09 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
 */
export const CAT_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서를 특정 차원(dimension)을 기준으로 결합(concatenate)할 때 사용하는 파라미터 구조체입니다. (What)
 * 텐서의 형태와 크기를 기반으로 각 텐서에서 어떤 위치의 값을 가져올지 인덱스를 계산하기 위해 존재합니다. (Why)
 */
struct Params {
  // 결합이 완료된 결과 텐서의 전체 요소(element) 개수입니다.
  size: u32,
  // X축 워크그룹 수. 2차원 그리드 인덱싱을 1차원 인덱스로 풀기 위한 변수입니다.
  workgroups_x: u32,
  // 결합하려는 축(axis)에서 첫 번째 텐서(A)가 차지하는 차원의 크기입니다.
  a_dim: u32,
  // 결합하려는 축(axis)에서 두 번째 텐서(B)가 차지하는 차원의 크기입니다.
  b_dim: u32,
  // 결합 축(axis)보다 하위에 있는 차원들의 요소 개수 곱입니다(Stride). 
  // 상위 차원이나 배치(batch)를 뛰어넘기 위한 보폭 역할을 합니다. (How)
  stride: u32,
  // 메모리 정렬(16바이트)을 위한 패딩 변수 1입니다.
  pad1: u32,
  // 메모리 정렬을 위한 패딩 변수 2입니다.
  pad2: u32,
  // 메모리 정렬을 위한 패딩 변수 3입니다.
  pad3: u32,
};

// params: 결합 연산에 필요한 차원 및 크기 정보를 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 결합될 첫 번째 입력 텐서 데이터 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 결합될 두 번째 입력 텐서 데이터 배열입니다 (읽기 전용).
@group(0) @binding(2) var<storage, read> b: array<f32>;
// out: A와 B가 이어진(Concatenated) 결과가 저장되는 배열입니다.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

/**
 * @function main
 * @brief 결과 텐서의 각 요소가 입력 텐서 A 혹은 B 중 어디서 와야 하는지를 계산하고 복사합니다. (What)
 * 병렬 인덱싱을 통하여 다차원 텐서의 결합 연산을 빠르게 수행하기 위해 만들어졌습니다. (Why)
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스 변수입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 파라미터에서 전체 데이터 개수를 가져옵니다.
  let num_elements = params.size;
  // 파라미터에서 X 방향 워크그룹 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  // 3차원 워크그룹 및 스레드 ID를 1차원 선형 인덱스로 변환합니다. (How)
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 계산된 인덱스가 전체 요소 개수를 넘어갈 경우 안전하게 함수를 종료합니다. (What)
  // 배열 범위를 벗어난 메모리에 대한 불법적인 쓰기를 방지하기 위해서입니다. (Why)
  if (idx >= num_elements) {
    return;
  }
  
  // 파라미터 구조체에서 내부 차원의 크기(stride)를 로드합니다.
  let stride = params.stride;
  // 파라미터 구조체에서 A 텐서의 결합 축 크기를 로드합니다.
  let a_dim = params.a_dim;
  // 파라미터 구조체에서 B 텐서의 결합 축 크기를 로드합니다.
  let b_dim = params.b_dim;
  
  // 결합된 이후 결과 텐서의 해당 축 길이를 계산합니다. (What)
  let out_dim_size = a_dim + b_dim;
  // 한 블록(결합 축 1개 단위 + 하위 차원 전체)이 차지하는 총 요소 개수(청크 크기)를 계산합니다. (How)
  let chunk_size = out_dim_size * stride;
  
  // 현재 1차원 인덱스가 어떤 배치(상위 차원들)에 속하는지 계산합니다. (How)
  let batch_idx = idx / chunk_size;
  // 현재 청크(chunk) 내에서 몇 번째 인덱스인지를 구합니다. (나머지 연산)
  let rem = idx % chunk_size;
  // 현재 청크 내에서 결합 축을 기준으로 몇 번째 위치에 있는지를 구합니다. (How)
  let dim_idx = rem / stride;
  // 결합 축보다 하위에 있는 차원에서 몇 번째 위치(stride_idx)인지를 구합니다.
  let stride_idx = rem % stride;
  
  // 현재 계산된 결합 축 상의 위치(dim_idx)가 텐서 A의 크기보다 작은지 검사합니다. (What)
  // 이 조건이 참이면 현재 요소는 텐서 A에서 가져와야 함을 의미합니다. (Why)
  if (dim_idx < a_dim) {
    // 텐서 A 배열 내부에서의 정확한 1차원 원본 인덱스를 복원 계산합니다. (How)
    // 배치 크기 * (A 차원 크기 * 스트라이드) + (A 안에서의 축 위치 * 스트라이드) + 하위 차원 오프셋
    let a_index = batch_idx * (a_dim * stride) + dim_idx * stride + stride_idx;
    // 계산된 인덱스를 사용해 텐서 A의 값을 결과 텐서에 복사합니다.
    out[idx] = a[a_index];
  } else {
    // dim_idx가 a_dim 이상이면 현재 요소는 텐서 B에서 가져와야 합니다.
    // 텐서 B의 내부 차원 인덱스로 변환하기 위해 A가 차지했던 크기를 뺍니다. (How)
    let b_dim_idx = dim_idx - a_dim;
    // 텐서 B 배열 내부에서의 원본 위치를 계산합니다. (How)
    let b_index = batch_idx * (b_dim * stride) + b_dim_idx * stride + stride_idx;
    // 계산된 인덱스를 사용해 텐서 B의 값을 결과 텐서에 복사합니다.
    out[idx] = b[b_index];
  }
}
`;
