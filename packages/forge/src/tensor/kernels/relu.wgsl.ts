/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const RELU_WGSL = `
// 구조체: Params
// 역할 (WHAT): ReLU(Rectified Linear Unit) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼(uniform) 데이터를 효율적으로 전달하고 메모리 정렬을 맞추기 위해 사용됩니다.
// 동작 방식 (HOW): 각 스레드가 처리할 전체 요소 개수와 2D 워크그룹 할당 정보를 메모리에서 읽어옵니다.
struct Params {
  // 변수: size
  // 역할: 입력 텐서가 가진 총 데이터 요소의 수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향으로 할당된 작업 그룹(workgroup)의 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 정렬을 맞추기 위한 패딩 변수
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체 값을 담고 있는 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: ReLU 활성화 함수가 적용될 원본 데이터를 가진 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: ReLU 연산 결과가 기록될 읽기/쓰기 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 각 요소에 대해 ReLU 활성화 함수(max(0, x))를 적용합니다.
// 목적 (WHY): 딥러닝 모델의 비선형성을 부여하기 위한 ReLU 연산을 GPU에서 병렬로 고속 처리하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 전역 ID를 활용하여 자신의 1D 인덱스를 계산한 후, 해당 인덱스에 있는 x의 값과 0 중 더 큰 값을 y에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 처리할 배열의 전체 요소 수를 유니폼 버퍼로부터 가져옵니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 글로벌 ID의 2D 인덱스를 1D 인덱스로 변환하기 위해 X축 워크그룹 수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: 현재 스레드가 처리해야 하는 1차원 데이터의 절대 인덱스
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 배열 크기 검사
  // 역할: 계산된 인덱스가 실제 데이터 범위(num_elements)를 벗어나는 경우 쓰레드 실행을 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 해당 위치 값을 0.0과 비교해 큰 값(음수는 0, 양수는 그대로)을 y 배열에 저장합니다.
  y[idx] = max(x[idx], 0.0);
}
`;
