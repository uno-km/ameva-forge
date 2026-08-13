/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const RELU_BACKWARD_WGSL = `
// 구조체: Params
// 역할 (WHAT): ReLU 역전파 연산에 필요한 메타데이터 정보를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 데이터를 전달하여 전체 연산 크기 등을 파악하게 합니다.
// 동작 방식 (HOW): 전체 크기(size)와 2D 그리드 변환을 위한 workgroups_x 인자를 넘겨줍니다.
struct Params {
  // 변수: size
  // 역할: 처리할 데이터의 전체 요소 수
  size: u32,
  // 변수: workgroups_x
  // 역할: x축 워크그룹의 총 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 메모리 정렬을 위한 패딩
  pad2: u32,
  pad3: u32,
}

// 변수: params
// 역할: 연산에 필요한 메타데이터가 담긴 유니폼 버퍼
@group(0) @binding(0) var<uniform> params : Params;

// 변수: X
// 역할: 순전파 시 입력되었던 원본 데이터를 담은 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> X : array<f32>;

// 변수: gradOutput
// 역할: 이전 레이어에서 흘러들어온 그래디언트(Gradient) 값을 담은 읽기 전용 버퍼
@group(0) @binding(2) var<storage, read> gradOutput : array<f32>;

// 변수: gradInput
// 역할: ReLU 연산의 역전파 결과로 계산된 그래디언트를 저장할 읽기/쓰기 버퍼
@group(0) @binding(3) var<storage, read_write> gradInput : array<f32>;

// 함수: main
// 역할 (WHAT): ReLU 역전파 그래디언트를 계산하는 컴퓨트 셰이더 메인 함수입니다.
// 목적 (WHY): 역전파 과정에서 X > 0 인 위치에만 그래디언트를 통과시키기 위해 존재합니다.
// 동작 방식 (HOW): 각 스레드는 1D 인덱스를 계산하고, X[index]의 값이 양수일 경우 gradOutput을 그대로 gradInput에 복사하고, 0 이하일 경우 0.0을 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // 변수: num_elements
  // 역할: 처리할 배열 요소의 총 개수를 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2D 인덱스를 1D 인덱스로 풀기 위해 가로 워크그룹 크기를 저장합니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: index
  // 역할: 현재 스레드의 작업을 가리키는 1차원 배열 위치 인덱스
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 데이터 경계 확인
  // 역할: 유효한 데이터 인덱스(num_elements 내부)인 경우에만 연산을 수행합니다.
  if (index < num_elements) {
    // 조건문: ReLU 미분 조건 (X > 0)
    // 역할: 원본 입력 값(X)이 양수인지 판단합니다.
    if (X[index] > 0.0) {
      // 변수 gradInput 갱신 (통과)
      // 역할: X가 양수였으므로 미분값이 1이 되어, 들어온 그래디언트를 그대로 전달합니다.
      gradInput[index] = gradOutput[index];
    } else {
      // 변수 gradInput 갱신 (차단)
      // 역할: X가 0 이하이므로 미분값이 0이 되어, 그래디언트 흐름을 0으로 차단합니다.
      gradInput[index] = 0.0;
    }
  }
}
`;
