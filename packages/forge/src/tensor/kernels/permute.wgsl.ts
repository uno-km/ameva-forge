/**
 * 생성일: 확인 불가 (Git 기록 없음 혹은 커밋 대기 상태)
 * 수정 이력:
 * - 특이사항 없음
 */
export const PERMUTE_WGSL = `
// 구조체: Params
// 역할 (WHAT): Permute(전치/축 교환) 연산에 필요한 차원, 형상(Shape) 및 보폭(Stride) 정보를 담은 구조체입니다.
// 목적 (WHY): 입력 텐서의 축을 지정된 순서대로 재배열하여 출력 텐서의 메모리 레이아웃을 계산하기 위해 유니폼 데이터를 전달합니다.
// 동작 방식 (HOW): rank와 총 요소 수를 제공하고, 최대 8차원을 지원하기 위해 vec4 두 개를 이어서 strides와 shape 정보를 제공합니다.
struct Params {
  // 변수: rank
  // 역할: 텐서가 가진 총 차원의 수
  rank: u32,
  // 변수: numElements
  // 역할: 텐서 내 존재하는 전체 데이터 요소의 개수
  numElements: u32,
  // 변수: workgroups_x
  // 역할: X축 방향으로 할당된 워크그룹(workgroup)의 총 개수
  workgroups_x: u32,
  // 변수: pad2
  // 역할: 16바이트 메모리 정렬을 위한 패딩 변수
  pad2: u32,
  // 변수: in_strides
  // 역할: 입력 텐서의 첫 4차원(0~3)에 대한 메모리 보폭
  in_strides: vec4<u32>,
  // 변수: in_strides_ext
  // 역할: 입력 텐서의 확장 4차원(4~7)에 대한 메모리 보폭
  in_strides_ext: vec4<u32>,
  out_shape: vec4<u32>,
  // 변수: out_shape_ext
  // 역할: 출력 텐서의 확장 4차원(4~7)에 대한 크기(Shape)
  out_shape_ext: vec4<u32>,
  // 변수: out_strides
  // 역할: 출력 텐서의 첫 4차원(0~3)에 대한 메모리 보폭
  out_strides: vec4<u32>,
  // 변수: out_strides_ext
  // 역할: 출력 텐서의 확장 4차원(4~7)에 대한 메모리 보폭
  out_strides_ext: vec4<u32>,
};

// 변수: params
// 역할: 셰이더 전역에서 접근 가능한 Permute 연산용 메타데이터 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 역할: 원본 데이터가 들어 있는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 역할: 축이 변환된 최종 데이터가 저장될 쓰기 가능한 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 출력 인덱스를 기반으로 다차원 좌표를 복원하고, 이를 입력 텐서의 보폭과 매칭하여 축 교환된 값을 저장합니다.
// 목적 (WHY): 텐서의 차원 순서를 바꾸는 연산(예: 행렬 전치, 채널 축 변경)을 GPU를 활용하여 병렬로 빠르게 수행하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 2D 기반 ID를 통해 출력 인덱스(out_idx)를 얻고, 반복문을 통해 각 차원별 인덱스를 분리해내어, 원본 보폭과 곱하여 in_idx를 도출해 값을 복사합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: out_idx
  // 역할: 2D 그리드 디스패치(dispatch)를 지원하기 위해 글로벌 ID x, y를 결합하여 만든 1차원 출력 인덱스
  // 주석: Compute global index supporting 2D grid dispatch
  let out_idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  
  // 조건문: 데이터 범위 초과 검사
  // 역할: 스레드의 계산된 인덱스가 전체 요소 크기를 넘어서는지 판단하여 초과 시 연산을 중지합니다.
  if (out_idx >= params.numElements) {
    return;
  }
  
  // 변수: out_idx_remaining
  // 역할: 각 차원의 좌표를 구하기 위해 나누기/나머지 연산을 하면서 변해가는 임시 나머지 인덱스 값
  var out_idx_remaining = out_idx;
  // 변수: in_idx
  // 역할: 입력 텐서에서 실제 데이터를 읽어올 1차원 메모리 인덱스의 누적 값
  var in_idx = 0u;
  
  // 반복문: for 루프 (모든 차원에 대한 순회)
  // 역할 (WHAT): 최상위 차원부터 시작하여 현재 차원에 해당하는 좌표를 구하고, 이를 바탕으로 원래 입력 배열의 인덱스를 계산합니다.
  // 목적 (WHY): 다차원 구조가 평면 배열(flat array)로 선형화되어 있으므로, 출력의 구조를 풀어 입력의 구조로 맵핑해야 하기 때문입니다.
  // 동작 방식 (HOW): 0부터 rank-1까지 순회하면서 차원(i)에 맞는 보폭(Stride) 값을 가져오고, 좌표(coord)를 구한 후 입력 인덱스를 누적합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: out_stride
    // 역할: 현재 루프 차원(i)에 해당하는 출력 텐서의 보폭
    var out_stride = 0u;
    // 변수: in_stride
    // 역할: 현재 루프 차원(i)에 해당하는 입력 텐서의 보폭
    var in_stride = 0u;
    
    // 조건문: 차원(i) 확인 및 보폭 할당
    // 역할 (WHAT): 루프 인덱스 i 값에 따라 vec4에 묶여 있는 각 차원의 보폭 값을 가져옵니다.
    // 목적 (WHY): WGSL에서는 배열 인덱싱을 지원하지 않는 vec4 구조체 필드에 동적으로 접근하기 위해 하드코딩 된 조건 분기가 필요하기 때문입니다.
    // 동작 방식 (HOW): i가 0~7 중 어느 것인지 확인하고, 해당하는 x, y, z, w 컴포넌트 값을 보폭 변수에 저장합니다.
    if (i == 0u) { out_stride = params.out_strides.x; in_stride = params.in_strides.x; }
    else if (i == 1u) { out_stride = params.out_strides.y; in_stride = params.in_strides.y; }
    else if (i == 2u) { out_stride = params.out_strides.z; in_stride = params.in_strides.z; }
    else if (i == 3u) { out_stride = params.out_strides.w; in_stride = params.in_strides.w; }
    else if (i == 4u) { out_stride = params.out_strides_ext.x; in_stride = params.in_strides_ext.x; }
    else if (i == 5u) { out_stride = params.out_strides_ext.y; in_stride = params.in_strides_ext.y; }
    else if (i == 6u) { out_stride = params.out_strides_ext.z; in_stride = params.in_strides_ext.z; }
    else if (i == 7u) { out_stride = params.out_strides_ext.w; in_stride = params.in_strides_ext.w; }
    
    // 변수: coord
    // 역할: 남은 1차원 인덱스를 출력 보폭으로 나누어 얻은 현재 차원(i)의 논리적 좌표값
    let coord = out_idx_remaining / out_stride;
    
    // 변수: out_idx_remaining 갱신
    // 역할: 다음 차원 계산을 위해 현재 차원에서 처리된 부분을 제외한 나머지(나머지 연산)를 저장합니다.
    out_idx_remaining = out_idx_remaining % out_stride;
    
    // 변수: in_idx 누적
    // 역할: 도출된 논리적 좌표(coord)에 원래 텐서의 보폭(in_stride)을 곱해, 원본 텐서에서 데이터를 읽어올 정확한 1차원 메모리 주소를 누적해 나갑니다.
    in_idx = in_idx + coord * in_stride;
  }
  
  // 변수 output 배열 쓰기
  // 역할: 매핑된 입력 텐서의 1차원 인덱스 위치(in_idx)에 있는 데이터를 읽어와 출력 텐서 위치(out_idx)에 복사하여 위치 바꿈(permute)을 완료합니다.
  output[out_idx] = input[in_idx];
}
`;
