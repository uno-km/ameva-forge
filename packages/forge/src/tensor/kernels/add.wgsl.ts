/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const ADD_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서의 덧셈 연산을 수행할 때 필요한 파라미터들을 담고 있는 구조체입니다.
 * GPU 내에서 uniform 버퍼를 통해 전달받아 연산의 크기나 차원을 제어하는 목적(Why)으로 사용됩니다.
 */
struct Params {
  // 전체 요소(element)의 개수입니다. 배열 범위를 초과하지 않도록 경계 검사를 수행하는 데(How) 사용됩니다.
  size: u32,
  // X 차원의 워크그룹 개수입니다. 2D 이상의 그리드에서 1차원 인덱스를 계산하기 위해(What) 필요합니다.
  workgroups_x: u32,
  // 16바이트 정렬(alignment) 규칙을 맞추기 위한 패딩 변수입니다. 특별한 로직을 수행하지는 않습니다.
  pad2: u32,
  // 16바이트 정렬(alignment) 규칙을 맞추기 위한 패딩 변수입니다.
  pad3: u32,
};

// params: GPU와 CPU 간의 데이터를 동기화하기 위한 Uniform 버퍼입니다. 연산에 필요한 메타데이터가 담겨 있습니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 첫 번째 입력 텐서의 데이터가 저장된 배열입니다. 읽기 전용(storage, read)으로 선언되었습니다.
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 두 번째 입력 텐서의 데이터가 저장된 배열입니다. 읽기 전용(storage, read)으로 선언되었습니다.
@group(0) @binding(2) var<storage, read> b: array<f32>;
// out: 연산 결과(덧셈)가 저장되는 출력 배열입니다. 읽고 쓰기가 가능한(storage, read_write) 형태로 선언되었습니다.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

/**
 * @function main
 * @brief WGSL의 메인 컴퓨트 셰이더 함수입니다. 두 배열 a와 b의 요소를 더하여 out에 저장합니다. (What)
 * 병렬 처리를 통해 대규모 텐서의 요소별(element-wise) 덧셈을 매우 빠르게 수행하기 위해(Why) 작성되었습니다.
 * 
 * @param global_id 내장 변수로, 현재 스레드가 전체 스레드 그리드에서 위치하는 3차원 인덱스입니다. (How)
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // params 구조체에서 연산할 전체 배열의 크기를 가져옵니다.
  let num_elements = params.size;
  // X축 방향으로 할당된 워크그룹의 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 2차원(혹은 3차원)으로 구성된 스레드 그리드의 인덱스를 1차원 배열 인덱스로 변환합니다. (How)
  // X인덱스에 Y인덱스 * (X방향 워크그룹 수 * 워크그룹 크기(64))를 더하여 선형 인덱스(idx)를 구합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 계산된 1차원 인덱스가 실제 처리해야 할 요소의 개수보다 작은지 검사합니다. (What)
  // 버퍼 오버플로우나 유효하지 않은 메모리 접근을 방지하기 위함입니다. (Why)
  if (idx < num_elements) {
    // a 배열과 b 배열의 동일한 인덱스 위치에 있는 값을 더하여, 그 결과를 out 배열에 저장합니다. (How)
    out[idx] = a[idx] + b[idx];
  }
}
`;
