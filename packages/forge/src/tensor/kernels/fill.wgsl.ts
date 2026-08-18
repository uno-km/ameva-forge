/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const FILL_WGSL = `
/**
 * 이 구조체(Params)는 텐서를 특정 값으로 채우기 위한 설정 정보를 담고 있습니다.
 * GPU에 유니폼 버퍼를 통해 전달되며, 16바이트 정렬을 맞추기 위해 패딩을 포함합니다.
 */
struct Params {
  numElements: u32, // 값을 채울 배열의 전체 요소 개수입니다.
  value: f32, // 배열을 채울 특정 단일 부동 소수점 값입니다.
  workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 개수입니다.
  pad2: u32, // 메모리 정렬을 위해 추가된 두 번째 패딩용 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에서 읽어들일 유니폼 데이터입니다.
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // 채워진 값이 쓰여질 출력 버퍼입니다.

/**
 * main 함수는 출력 배열의 모든 요소에 지정된 값을 병렬로 기록합니다.
 * 텐서를 특정 상수값으로 초기화하는 fill 연산을 GPU에서 고속으로 수행하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.numElements; // 전체 요소 개수를 유니폼 변수에서 가져옵니다.
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
  
  // 계산된 인덱스가 전체 배열 크기보다 크거나 같다면 작업을 수행하지 않고 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 지정된 인덱스 위치에 설정된 상수 값(params.value)을 저장합니다.
  output[idx] = params.value;
}
`;
