/**
 * 생성일: 2026-08-12T12:23:09+09:00
 * 수정 이력:
 * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 */
export const PAD_WGSL = `
// 구조체: Params
// 역할 (WHAT): 텐서 패딩 연산에 필요한 모든 형태(Shape), 보폭(Stride), 설정 변수들을 담고 있는 구조체입니다.
// 목적 (WHY): 패딩 된 새로운 텐서를 생성하기 위해 원본 텐서의 좌표와 출력 텐서의 좌표를 매핑하는 데 필요한 정보를 제공하기 위함입니다.
// 동작 방식 (HOW): 각 차원에 대한 크기, 원본/출력 메모리 보폭 정보, 추가할 패딩 값 등을 참조하여 변환된 인덱스를 계산합니다.
struct Params {
  // 변수: num_elements
  // 역할: 패딩이 완료된 최종 출력 텐서의 전체 요소 개수
  num_elements: u32,
  // 변수: rank
  // 역할: 텐서의 차원(Rank) 수
  rank: u32,
  // 변수: pad_val
  // 역할: 빈 공간에 채워 넣을 상수 값(패딩 값)
  pad_val: f32,
  // 변수: _pad
  // 역할: WebGPU 메모리 정렬(16바이트)을 맞추기 위한 여분(padding) 변수
  _pad: u32,
  // 변수: in_strides
  // 역할: 최대 8차원까지 지원하는 원본 입력 텐서의 차원별 메모리 보폭(Stride) 배열
  in_strides: array<u32, 8>,
  // 변수: out_strides
  // 역할: 패딩 적용 후 출력 텐서의 차원별 메모리 보폭 배열
  out_strides: array<u32, 8>,
  // 변수: pad_before
  // 역할: 각 차원의 앞부분(before)에 추가되는 패딩의 크기를 저장하는 배열
  pad_before: array<u32, 8>,
  // 변수: in_shape
  // 역할: 입력 텐서의 원래 차원별 크기(Shape)를 저장하는 배열
  in_shape: array<u32, 8>,
};

// 변수: params
// 역할: 패딩 연산의 메타데이터를 저장하는 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 역할: 패딩 되기 전의 원본 데이터가 저장되어 있는 스토리지 버퍼
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 역할: 패딩 된 결과 데이터가 기록될 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 출력 텐서의 각 인덱스를 기준으로 원본 인덱스를 역추적하여 값을 복사하거나 패딩 값을 채웁니다.
// 목적 (WHY): 입력 배열 주변에 원하는 크기와 값으로 여백(패딩)을 추가하여 크기가 확장된 텐서를 반환하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 출력 1D 인덱스(idx)를 받아 다차원 좌표(coord)로 변환한 후, 이 좌표가 원본 텐서 영역 내인지 확인하여 원본 값을 쓰거나 pad_val을 채웁니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: idx
  // 역할: 현재 스레드가 담당하는 출력 텐서의 1차원 전역 인덱스
  let idx = global_id.x;
  
  // 조건문: 인덱스 범위 확인
  // 역할: idx가 결과 텐서의 전체 크기를 넘어서면 실행을 중지하여 유효하지 않은 메모리 접근을 방지합니다.
  if (idx >= params.num_elements) { return; }

  // 변수: temp
  // 역할: 1차원 인덱스를 다차원 좌표로 분해할 때 남은 인덱스 값을 저장 및 갱신하기 위한 임시 변수
  var temp = idx;
  // 변수: in_idx
  // 역할: 역계산된 원본 텐서의 1차원 인덱스를 누적할 변수
  var in_idx = 0u;
  // 변수: in_bounds
  // 역할: 현재 계산하는 출력 좌표가 원본 텐서의 범위 안에 속해 있는지를 나타내는 불리언 플래그
  var in_bounds = true;

  // 반복문: for 루프 (차원 탐색)
  // 역할 (WHAT): 최고 차원부터 최하 차원까지 각 차원의 좌표를 구하고, 이를 이용해 원본 입력 텐서의 플랫(flat) 인덱스를 누적 연산합니다.
  // 목적 (WHY): N차원(최대 8차원) 데이터를 1차원 배열로 평탄화(Flatten)한 메모리 구조에서 정확한 매핑을 계산하기 위함입니다.
  // 동작 방식 (HOW): 나누기와 나머지 연산을 사용해 현재 차원의 좌표(coord)를 구한 뒤, 원본 텐서 구간(pad_before ~ pad_before + in_shape)에 속하는지 검사합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: coord
    // 역할: 출력 텐서의 i번째 차원에 대한 구체적 좌표(인덱스)
    let coord = temp / params.out_strides[i];
    
    // 변수: temp 갱신
    // 역할: 다음 하위 차원 계산을 위해 남은 나머지 값을 임시 변수에 대입합니다.
    temp = temp % params.out_strides[i];
    
    // 조건문: 원본 영역 이탈 확인
    // 역할: 계산된 해당 차원의 좌표가 패딩 영역(원본 데이터가 없는 곳)인지 판단합니다.
    if (coord < params.pad_before[i] || coord >= params.pad_before[i] + params.in_shape[i]) {
      // 변수: in_bounds 갱신
      // 역할: 영역 바깥이므로 in_bounds를 false로 변경하고 루프를 탈출합니다.
      in_bounds = false;
      break;
    }
    
    // 변수: in_coord
    // 역할: 출력 텐서 좌표에서 앞부분 패딩(pad_before)을 빼서 원본 텐서 기준의 순수 좌표를 구합니다.
    let in_coord = coord - params.pad_before[i];
    
    // 변수: in_idx 누적
    // 역할: 구한 원본 좌표에 해당 차원의 보폭(in_strides)을 곱하여 1D 원본 인덱스를 점진적으로 계산합니다.
    in_idx = in_idx + in_coord * params.in_strides[i];
  }

  // 조건문: 값 삽입 결정
  // 역할: 구해진 플래그(in_bounds)를 바탕으로 배열에 원본 데이터를 쓸지, 패딩 값을 쓸지 분기합니다.
  if (in_bounds) {
    // 변수 output 갱신 (원본)
    // 역할: 출력 배열에 입력 배열의 데이터를 그대로 복사합니다.
    output[idx] = input[in_idx];
  } else {
    // 변수 output 갱신 (패딩)
    // 역할: 출력 배열에 미리 설정해 둔 패딩 상수 값(pad_val)을 삽입합니다.
    output[idx] = params.pad_val;
  }
}
`;
