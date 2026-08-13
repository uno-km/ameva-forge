/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const EXP_WGSL = `
/**
 * 이 구조체(Params)는 워크그룹과 데이터의 크기를 설정하기 위해 존재합니다.
 * 패딩 변수들은 WebGPU 버퍼의 16바이트 정렬 규칙을 준수하기 위해 사용됩니다.
 */
struct Params {
  size: u32, // 처리해야 할 전체 요소의 개수입니다.
  workgroups_x: u32, // X축 방향으로 스패닝된 워크그룹의 총 개수입니다.
  pad2: u32, // 메모리 정렬을 위해 존재하는 사용되지 않는 패딩 변수입니다.
  pad3: u32, // 메모리 정렬을 위해 존재하는 사용되지 않는 패딩 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 상수 파라미터입니다.
@group(0) @binding(1) var<storage, read> x: array<f32>; // 읽기 전용으로 설정된 입력 텐서 데이터입니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // 연산 결과가 쓰여질 출력 텐서 데이터입니다.

/**
 * main 함수는 각 텐서 요소에 대해 자연 상수 e를 밑으로 하는 지수 함수(exp) 연산을 수행합니다.
 * 이 함수가 존재하는 이유는 텐서의 모든 원소에 대해 병렬적으로 지수 연산을 처리하기 위함입니다.
 * GPU의 각 스레드는 고유한 global_id를 받아 배열 내 자신의 작업 위치를 계산하고 결과를 출력 버퍼에 저장합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size; // 전체 계산해야 하는 원소의 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x; // 3D 그리드 기반의 1D 인덱스 계산을 위해 x축 워크그룹 수를 가져옵니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u; // 2D 형태로 스패닝된 글로벌 ID를 1D 인덱스로 변환하여 현재 스레드가 처리할 데이터의 위치를 구합니다.
  
  // 현재 스레드의 인덱스가 전체 배열 크기를 초과하면, 더 이상 처리하지 않고 함수를 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 계산된 인덱스의 입력값 x[idx]에 대해 지수 함수를 적용한 뒤, 그 결과를 출력 배열 y의 동일한 위치에 저장합니다.
  y[idx] = exp(x[idx]);
}
`;
