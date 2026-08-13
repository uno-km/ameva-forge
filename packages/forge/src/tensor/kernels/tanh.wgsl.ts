/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const TANH_WGSL = `
// 구조체: Params
// 목적: Tanh 커널 연산 시 필요한 설정값을 제공합니다.
// 작동 방식: 배열의 전체 요소 수와 워크그룹의 X축 크기를 포함합니다.
struct Params {
  // 변수: size
  // 목적: 전체 연산 대상 원소의 개수를 저장합니다.
  // 작동 방식: 범위를 초과하는 메모리 접근을 방지하는 기준으로 쓰입니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 워크그룹 수를 저장합니다.
  // 작동 방식: 3차원 스레드 ID를 1차원 인덱스로 변환할 때 곱해집니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 유니폼 구조체의 메모리 오프셋 규칙을 준수합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 유니폼 구조체의 메모리 오프셋 규칙을 준수합니다.
  pad3: u32,
};

// 변수: params
// 목적: 커널의 설정값을 가지고 있는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0을 통해 GPU에 전달되어 읽기 전용으로 참조됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 목적: Tanh 함수를 적용할 입력 텐서 데이터를 담고 있는 버퍼입니다.
// 작동 방식: 바인딩 1에 매핑되며 원본 데이터를 제공합니다.
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 목적: Tanh 함수의 계산 결과를 저장할 출력 버퍼입니다.
// 작동 방식: 바인딩 2에 매핑되며 계산된 활성화 값이 각 인덱스에 저장됩니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 목적: 병렬 스레드를 이용하여 배열의 각 요소에 대해 Tanh(쌍곡탄젠트) 함수를 계산합니다.
// 작동 방식: 내장 함수인 tanh()를 호출하여 y 배열에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 배열의 전체 요소 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 size를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기입니다.
  // 작동 방식: 유니폼 버퍼에서 workgroups_x를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 스레드가 처리할 1차원 배열의 인덱스입니다.
  // 작동 방식: global_id 정보를 바탕으로 평면화된 인덱스를 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 인덱스 범위를 초과한 스레드가 실행되는 것을 막습니다.
  // 작동 방식: idx가 num_elements 이상일 경우 함수를 조기 종료(return)합니다.
  if (idx >= num_elements) {
    return;
  }

  // 연산: y[idx] 기록
  // 목적: 특정 요소의 Tanh 값을 계산하여 저장합니다.
  // 작동 방식: WGSL 내장 함수 tanh(x[idx])를 호출하고 그 결과를 y[idx]에 씁니다.
  y[idx] = tanh(x[idx]);
}
`;
