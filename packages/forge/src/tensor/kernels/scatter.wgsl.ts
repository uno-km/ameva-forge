/**
 * 생성일: 2026-08-12T12:23:09+09:00
 * 수정 이력:
 * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 */
export const SCATTER_WGSL = `
// 구조체: Params
// 역할 (WHAT): 스캐터(Scatter) 연산에 필요한 차원, 보폭 및 타겟 축 정보를 저장하는 구조체입니다.
// 목적 (WHY): 입력 데이터 텐서와 인덱스 텐서를 조합하여 출력 텐서의 어느 위치에 값을 기록할지 결정하기 위함입니다.
// 동작 방식 (HOW): 각 차원에 대한 크기(rank), 흩뿌릴 차원(dim), 인덱스/입력의 보폭 정보를 읽어와 좌표를 계산합니다.
struct Params {
  // 변수: num_elements
  // 역할: 처리할 입력 요소들의 전체 개수
  num_elements: u32,
  // 변수: dim
  // 역할: 인덱스 값으로 대체되어 흩뿌려질 대상 차원 축
  dim: u32,
  // 변수: rank
  // 역할: 텐서가 갖는 전체 차원 수
  rank: u32,
  // 변수: workgroups_x
  // 역할: 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수
  workgroups_x: u32,
  // 변수: x_strides
  // 역할: 출력 배열(입력과 동일한 형상을 가지는 베이스)의 각 차원별 메모리 보폭 배열
  x_strides: array<u32, 8>,
  // 변수: idx_strides
  // 역할: 인덱스 텐서의 각 차원별 메모리 보폭 배열
  idx_strides: array<u32, 8>,
  // 변수: x_shape
  // 역할: 출력 텐서의 각 차원별 크기 배열 (인덱스 바운드 검사용)
  x_shape: array<u32, 8>,
};

// 변수: params
// 역할: 스캐터 연산의 메타데이터를 담은 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: index
// 역할: 흩뿌릴 위치 정보를 가지고 있는 인덱스 배열(읽기 전용 정수 스토리지 버퍼)
@group(0) @binding(1) var<storage, read> index: array<u32>;

// 변수: src
// 역할: 출력 배열에 복사할 원본 값을 가지고 있는 소스 배열(읽기 전용 스토리지 버퍼)
@group(0) @binding(2) var<storage, read> src: array<f32>;

// 변수: output
// 역할: 원본 값들이 인덱스 배열의 지시에 따라 흩뿌려진 최종 결과물이 저장될 스토리지 버퍼
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 주어진 인덱스 텐서의 값에 따라 소스 데이터를 출력 텐서의 특정 위치에 저장합니다.
// 목적 (WHY): 특정 차원의 값을 인덱스로 치환하여(Scatter-Elements) 텐서 내 원하는 위치에 데이터를 쓰기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 1차원 ID를 다차원 좌표로 변환하고, 지정된 축(dim)에 대해서만 원래 좌표 대신 인덱스 텐서의 값을 좌표로 사용하여 출력 위치를 정합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: idx
  // 역할: 소스 데이터와 인덱스 데이터의 1차원 메모리 인덱스
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  
  // 조건문: 데이터 경계 검사
  // 역할: 할당된 스레드의 인덱스가 전체 크기(num_elements)를 초과하는지 검사합니다.
  if (idx >= params.num_elements) { return; }

  // 변수: temp
  // 역할: 다차원 좌표로 분리해 나가기 위해 남은 인덱스 수치를 보관하는 임시 변수
  var temp = idx;
  // 변수: out_idx
  // 역할: 최종적으로 계산된 출력 배열의 1차원 메모리 인덱스를 누적할 변수
  var out_idx = 0u;

  // 반복문: for 루프 (모든 차원 순회)
  // 역할 (WHAT): 최상위 차원부터 0번째 차원까지 각 차원의 좌표를 구하고, 이를 이용해 출력 인덱스를 계산합니다.
  // 목적 (WHY): 1차원 인덱스를 다시 N차원 좌표로 풀고, 특정 차원(dim)에 대해서만 값을 교체하기 위해 필요합니다.
  // 동작 방식 (HOW): i가 dim과 같을 경우, 계산된 논리적 좌표 대신 index 배열에 있는 값을 가져와서 보폭을 곱하고, 그 외의 경우 원래 좌표에 보폭을 곱합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: coord
    // 역할: 현재 차원(i)에 해당하는 인덱스 텐서 기준의 다차원 논리 좌표
    let coord = temp / params.idx_strides[i];
    
    // 변수: temp 갱신
    // 역할: 다음 차원 계산을 위해 나머지 값을 임시 변수에 업데이트합니다.
    temp = temp % params.idx_strides[i];
    
    // 조건문: 타겟 차원(dim) 여부 검사
    // 역할: 현재 처리 중인 차원이 인덱스 값으로 대체할 타겟 차원인지 판단합니다.
    if (i == params.dim) {
      let raw_bits = index[idx];
      let dim_size = i32(params.x_shape[i]);
      var signed_idx = bitcast<i32>(raw_bits);
      if (signed_idx < 0) {
        signed_idx = signed_idx + dim_size;
      }
      // OOB check: skip execution if index is out of bounds
      if (signed_idx < 0 || signed_idx >= dim_size) {
        return;
      }
      let valid_idx = u32(signed_idx);
      out_idx = out_idx + valid_idx * params.x_strides[i];
    } else {
      // 변수: out_idx 누적 (일반 축)
      // 역할: 원래 논리적 좌표(coord)에 출력 보폭(x_strides)을 곱해 더합니다.
      out_idx = out_idx + coord * params.x_strides[i];
    }
  }

  // 주석: 엄밀한 원자성(Atomic)은 제공하지 않지만 인덱스가 겹치지 않는 단순 스캐터의 경우 정상 동작함.
  // 변수 output 갱신
  // 역할: 치환이 완료되어 도출된 출력 인덱스 위치(out_idx)에 소스 배열의 데이터(src[idx])를 저장합니다.
  // Not strictly atomic, but for simple scatter where indices are unique it's fine.
  output[out_idx] = src[idx];
}
`;
