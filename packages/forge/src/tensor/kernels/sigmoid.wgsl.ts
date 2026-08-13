/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const SIGMOID_WGSL = `
// 구조체: Params
// 역할 (WHAT): 시그모이드(Sigmoid) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 인자를 넘겨주어 전체 요소 수 등의 전역 설정을 공유하기 위함입니다.
// 동작 방식 (HOW): 요소 크기와 2D 워크그룹 할당 정보를 메모리 정렬을 맞추어 전달합니다.
struct Params {
  // 변수: size
  // 역할: 처리 대상 배열이 가진 전체 원소의 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 워크그룹 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 메모리 정렬(alignment)용 패딩
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체를 담고 있는 유니폼 버퍼 (바인딩 0)
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: 시그모이드 활성화 함수가 적용될 원본 데이터가 저장된 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: 시그모이드 연산 결과가 기록될 읽기/쓰기 가능한 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 각 요소에 대해 시그모이드 활성화 함수(1 / (1 + exp(-x)))를 적용합니다.
// 목적 (WHY): 신경망의 값을 0과 1 사이로 변환하는 활성화 함수 연산을 GPU에서 병렬로 고속 수행하기 위함입니다.
// 동작 방식 (HOW): 64크기의 워크그룹 내 스레드들이 1D 인덱스를 계산하고, 범위를 초과하지 않으면 수학 함수 exp를 이용해 시그모이드 수식을 계산 후 y 배열에 씁니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 버퍼에서 배열의 총 원소 개수를 읽어옵니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2차원 워크그룹 인덱스를 1차원 인덱스로 변환하기 위해 X축 워크그룹 수를 읽어옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: 현재 쓰레드가 처리해야 할 1차원 데이터 인덱스
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 배열 경계 확인
  // 역할: 인덱스가 실제 배열의 범위를 벗어날 경우 셰이더 실행을 조기 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 값에 시그모이드 공식을 적용한 결과를 y 배열에 저장합니다.
  y[idx] = 1.0 / (1.0 + exp(-x[idx]));
}
`;
