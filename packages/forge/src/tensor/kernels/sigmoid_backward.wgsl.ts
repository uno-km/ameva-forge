/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const SIGMOID_BACKWARD_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터들을 정의합니다. 메모리 정렬을 위해 패딩 변수가 포함되어 있습니다.
// 작동 방식: size와 workgroups_x 정보를 포함하여 작업 스레드가 자신의 위치를 파악할 수 있게 합니다.
struct Params {
  // 변수: size
  // 목적: 처리해야 할 전체 요소의 총 개수를 저장합니다.
  // 작동 방식: 배열의 범위를 초과하는 접근을 방지하는 기준값으로 사용됩니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수를 저장합니다.
  // 작동 방식: 2차원 워크그룹 인덱스를 1차원 전역 인덱스로 변환할 때 곱해지는 계수로 사용됩니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 접근 성능을 최적화하고 데이터 구조의 규격을 맞추는 역할을 합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 접근 성능을 최적화하고 데이터 구조의 규격을 맞추는 역할을 합니다.
  pad3: u32,
};

// 변수: params
// 목적: 외부에서 전달되는 설정값들을 저장하는 유니폼 버퍼(Uniform buffer) 변수입니다.
// 작동 방식: 바인딩 그룹 0, 바인딩 0에 매핑되어 워크그룹 실행 시 필요한 메타데이터를 제공합니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: grad
// 목적: 역전파(Backpropagation) 단계에서 이전 층(layer)으로부터 전달받은 손실(loss)의 기울기(gradient)를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 1에 매핑되며, 최종 기울기를 계산할 때 곱해지는 입력값으로 쓰입니다.
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// 변수: sigmoid_output
// 목적: 순전파(Forward propagation) 단계에서 미리 계산되었던 Sigmoid 함수의 출력 결과를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 2에 매핑되며, Sigmoid 미분 공식을 적용하기 위한 상태값으로 사용됩니다.
@group(0) @binding(2) var<storage, read> sigmoid_output: array<f32>;

// 변수: output
// 목적: 계산된 Sigmoid 함수의 역전파 기울기 결과를 저장할 읽기/쓰기 가능 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 3에 매핑되며, 각 스레드에서 계산된 최종 미분값이 이곳에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: Sigmoid 함수의 역전파(Backward) 연산을 병렬로 수행하는 메인 컴퓨트 셰이더(Compute Shader) 진입점입니다.
// 작동 방식: Sigmoid 미분 공식인 'sigmoid_output * (1 - sigmoid_output)'을 사용하여 이전 기울기 'grad'와 곱한 뒤 최종 기울기를 계산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산해야 할 총 원소의 개수를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 size 필드를 읽어와 저장합니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 workgroups_x 필드를 읽어와 저장합니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 실행 중인 스레드가 담당할 1차원 데이터 인덱스를 계산합니다.
  // 작동 방식: 3차원인 global_id 값을 바탕으로, Y축 인덱스에 (X축 워크그룹 개수 * 워크그룹 크기 64)를 곱하고 X축 인덱스를 더해 평면화(flatten)된 인덱스를 구합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 범위를 벗어난 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: 계산된 인덱스(idx)가 처리해야 할 전체 요소 수(num_elements) 이상인지 확인합니다.
  if (idx >= num_elements) {
    // 유효 범위를 초과하면 아무 연산도 수행하지 않고 함수를 종료합니다.
    return;
  }

  // 연산: output[idx] 갱신
  // 목적: 최종적으로 입력 노드에 전달할 기울기(Gradient) 값을 도출하여 저장합니다.
  // 작동 방식: 체인 룰(Chain rule)에 의해 '상류에서 온 기울기(grad[idx])' * '로컬 미분값(sigmoid_output[idx] * (1.0 - sigmoid_output[idx]))'을 연산한 후 배열에 기록합니다.
  output[idx] = grad[idx] * sigmoid_output[idx] * (1.0 - sigmoid_output[idx]);
}
`;
