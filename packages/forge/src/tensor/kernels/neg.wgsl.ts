/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const NEG_WGSL = `
// 구조체: Params
// 역할 (WHAT): 음수화(Negation) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 데이터를 전달하고 16바이트 정렬 규칙을 준수하기 위해 정의되었습니다.
// 동작 방식 (HOW): 셰이더가 실행될 때 전체 데이터 크기(size)와 2D 기반 분할 시 사용되는 x축 워크그룹 크기를 참조합니다.
struct Params {
  // 변수: size
  // 역할: 처리할 전체 데이터 배열의 요소 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 작업 그룹 수 (2D 그리드 인덱싱에 사용)
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 구조체의 메모리 정렬을 위한 여분(padding) 공간
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체를 저장하는 유니폼 버퍼 (인덱스 바인딩 0)
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: 입력 데이터를 담는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: 계산 결과(음수화된 값)가 저장될 읽기/쓰기 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 모든 요소에 대해 부호를 반전시키는 메인 컴퓨트 함수입니다.
// 목적 (WHY): GPU의 병렬 아키텍처를 활용하여 빠르고 동시적인 부호 반전 연산을 수행하기 위함입니다.
// 동작 방식 (HOW): 64 워크그룹 사이즈 내에서 각 스레드가 전역 ID를 사용해 1D 인덱스를 계산하고, 유효한 범위 내에서 - 연산을 적용합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 인자를 통해 전달된 전체 배열 크기를 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2D 그리드 맵핑을 풀기 위한 가로(X축) 워크그룹의 개수를 저장합니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: x 및 y 방향 워크그룹 ID와 로컬 ID를 결합하여 처리할 1D 데이터 인덱스를 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: out-of-bounds 방지
  // 역할: 계산된 인덱스가 실제 데이터의 요소 개수를 초과하면 처리를 조기 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 해당 인덱스 값을 읽어와 음수 기호를 붙인 후 y 배열에 씁니다.
  y[idx] = -x[idx];
}
`;
