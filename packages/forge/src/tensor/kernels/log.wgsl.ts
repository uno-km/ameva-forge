/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const LOG_WGSL = `
/**
 * 이 구조체(Params)는 워크그룹과 데이터의 크기를 설정하기 위해 존재합니다.
 * 패딩 변수들은 WebGPU 버퍼의 16바이트 정렬 규칙을 준수하기 위해 사용됩니다.
 */
struct Params {
  size: u32, // 처리해야 할 텐서의 전체 원소 개수입니다.
  workgroups_x: u32, // X축을 따라 생성된 워크그룹의 총 개수입니다.
  pad2: u32, // 16바이트 메모리 정렬을 위해 남겨둔 미사용 변수입니다.
  pad3: u32, // 16바이트 메모리 정렬을 위해 남겨둔 미사용 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 메타데이터 및 설정값이 담긴 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> x: array<f32>; // 자연로그 연산을 수행할 대상이 되는 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // 자연로그 연산 결과가 저장될 출력 텐서입니다.

/**
 * main 함수는 입력 텐서의 각 요소에 대하여 자연로그(log) 연산을 수행합니다.
 * 요소별(element-wise) 자연로그 연산을 GPU의 병렬 처리 능력을 통해 가속화하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size; // 텐서의 전체 원소 개수를 유니폼 변수로부터 가져옵니다.
  let workgroups_x = params.workgroups_x; // 1차원 인덱스로 변환하기 위해 X축 워크그룹 크기를 가져옵니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u; // 2D 형태의 global_id를 1차원 평면 인덱스로 펼쳐서 현재 스레드의 작업 위치를 결정합니다.
  
  // 계산된 현재 스레드의 인덱스가 전체 텐서 크기를 벗어나면 작업을 수행하지 않고 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 현재 인덱스에 해당하는 입력 텐서 값에 대해 내장 함수 log()를 호출하고, 그 결과를 출력 텐서의 동일 위치에 저장합니다.
  y[idx] = log(x[idx]);
}
`;
