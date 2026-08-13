/**
 * 생성일: 2026-08-12T12:14:52+09:00
 * 수정 이력:
 * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const MUL_WGSL = `
// 구조체: Params
// 역할 (WHAT): 곱셈 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼(uniform) 데이터를 효율적으로 전달하기 위해 존재합니다.
// 동작 방식 (HOW): 각 스레드가 처리해야 할 전체 크기 및 작업 그룹 설정 값을 메모리에서 읽어옵니다.
struct Params {
  // 변수: size
  // 역할: 텐서 내 처리할 전체 요소의 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 작업 그룹(workgroup) 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 정렬(alignment)을 맞추기 위한 패딩 변수
  pad2: u32,
  pad3: u32,
}

// 변수: params
// 역할: Params 구조체 타입의 유니폼 버퍼 바인딩
@group(0) @binding(0) var<uniform> params : Params;

// 변수: A
// 역할: 첫 번째 입력 텐서 데이터를 담고 있는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> A : array<f32>;

// 변수: B
// 역할: 두 번째 입력 텐서 데이터를 담고 있는 읽기 전용 스토리지 버퍼
@group(0) @binding(2) var<storage, read> B : array<f32>;

// 변수: C
// 역할: 곱셈 결과가 저장될 읽기/쓰기 가능한 출력 스토리지 버퍼
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

// 함수: main
// 역할 (WHAT): 두 텐서 A와 B의 각 요소를 곱하여 C에 저장하는 컴퓨트 셰이더 메인 함수입니다.
// 목적 (WHY): 병렬 처리를 통해 대규모 배열의 요소별 곱셈(Element-wise multiplication)을 빠르게 수행하기 위함입니다.
// 동작 방식 (HOW): 64개의 스레드를 가진 작업 그룹에서 실행되며, 1차원 배열을 2D 그리드 방식으로 매핑하여 현재 스레드의 글로벌 인덱스를 계산하고, 이 인덱스가 전체 크기(size)보다 작을 때만 곱셈을 수행합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 버퍼에서 전체 요소의 개수를 가져와 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: X축 방향의 작업 그룹 크기를 가져와 저장합니다. (2D 그리드를 1D 인덱스로 변환할 때 사용)
  let workgroups_x = params.workgroups_x;
  
  // 변수: index
  // 역할: 현재 스레드가 처리해야 할 1차원 글로벌 데이터 인덱스를 계산합니다.
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: index 유효성 검사
  // 역할: 인덱스가 실제 데이터 크기(num_elements) 배열 범위 내에 있는지 확인합니다.
  if (index < num_elements) {
    // 변수 C 배열 갱신
    // 역할: A 배열과 B 배열의 동일 인덱스 위치에 있는 값을 곱하여 C 배열에 저장합니다.
    C[index] = A[index] * B[index];
  }
}
`;
