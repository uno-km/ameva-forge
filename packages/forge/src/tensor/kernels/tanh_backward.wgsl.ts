/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const TANH_BACKWARD_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터들을 정의합니다. 메모리 정렬을 위해 패딩 변수가 포함되어 있습니다.
// 작동 방식: 배열 크기(size)와 2차원 워크그룹의 X축 크기(workgroups_x)를 제공합니다.
struct Params {
  // 변수: size
  // 목적: 연산할 전체 배열 요소의 개수입니다.
  // 작동 방식: 배열 범위를 초과하는 인덱스 접근을 차단하기 위한 경계값으로 쓰입니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향 워크그룹의 개수입니다.
  // 작동 방식: 3D 워크그룹 인덱스를 1D 전역 인덱스로 변환할 때 필요합니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad3: u32,
};

// 변수: params
// 목적: 커널의 설정값을 가지고 있는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0을 통해 GPU에 전달되어 읽기 전용으로 참조됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: grad
// 목적: 역전파 시 이전 층(상류)으로부터 전달받은 손실 기울기를 저장하는 버퍼입니다.
// 작동 방식: 바인딩 1에 매핑되어 최종 미분값 계산에 곱해지는 입력값으로 쓰입니다.
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// 변수: tanh_output
// 목적: 순전파 단계에서 이미 계산되었던 Tanh 함수의 출력 결과를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 2에 매핑되며, Tanh 미분 공식을 적용하기 위한 상태값으로 사용됩니다.
@group(0) @binding(2) var<storage, read> tanh_output: array<f32>;

// 변수: output
// 목적: 계산된 Tanh 함수의 역전파 기울기 결과를 저장할 읽기/쓰기 가능 버퍼입니다.
// 작동 방식: 바인딩 3에 매핑되며, 각 스레드에서 계산된 최종 미분값이 이곳에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: Tanh 함수의 역전파(Backward) 연산을 병렬로 수행하는 메인 컴퓨트 셰이더 진입점입니다.
// 작동 방식: Tanh 미분 공식 '1 - tanh_output^2'에 이전 기울기 'grad'를 곱하여 최종 기울기를 계산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산해야 할 총 원소의 개수를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 size 필드를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 workgroups_x 필드를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 실행 중인 스레드가 담당할 1차원 데이터 인덱스를 계산합니다.
  // 작동 방식: 3차원인 global_id 값을 바탕으로, Y축 인덱스에 (X축 워크그룹 개수 * 64)를 곱하고 X축 인덱스를 더해 평면화합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 범위를 벗어난 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: idx가 num_elements 이상인지 확인하여 맞으면 함수를 종료합니다.
  if (idx >= num_elements) {
    return;
  }

  // 연산: output[idx] 갱신
  // 목적: 최종적으로 입력 노드에 전달할 기울기(Gradient) 값을 도출하여 저장합니다.
  // 작동 방식: 체인 룰(Chain rule)에 의해 '상류에서 온 기울기(grad[idx])' * '로컬 미분값(1.0 - tanh_output[idx] * tanh_output[idx])'을 계산해 기록합니다.
  output[idx] = grad[idx] * (1.0 - tanh_output[idx] * tanh_output[idx]);
}
`;
