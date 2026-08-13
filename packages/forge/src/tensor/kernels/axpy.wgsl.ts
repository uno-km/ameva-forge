/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const AXPY_WGSL = `
/**
 * @struct Params
 * @brief AXPY (Y = Y - alpha * X 형태, 여기서는 Parameter Update) 연산의 파라미터를 담고 있습니다. (What)
 * 머신러닝의 경사하강법(Gradient Descent) 시 가중치를 학습률(learning rate)에 비례하여 업데이트하기 위해 존재합니다. (Why)
 */
struct Params {
  // 업데이트를 수행할 전체 요소(가중치/파라미터)의 개수입니다.
  numElements: u32,
  // 학습률(learning rate, lr)입니다. 그레이디언트(grad)가 파라미터에 미치는 영향을 조절합니다.
  lr: f32,
  // 16바이트(float4) 정렬을 맞추기 위해 사용되는 패딩입니다.
  pad1: u32,
  // 16바이트(float4) 정렬을 맞추기 위해 사용되는 패딩입니다.
  pad2: u32,
};

// params: 균일한 크기를 가지는 파라미터 버퍼입니다. 업데이트에 필요한 총 요소 수와 학습률 등을 포함합니다.
@group(0) @binding(0) var<uniform> params: Params;
// grad: 파라미터에 대한 기울기(Gradient) 값을 담고 있는 배열입니다. (읽기 전용)
@group(0) @binding(1) var<storage, read> grad: array<f32>;
// param: 현재 모델의 가중치(파라미터) 배열입니다. 읽고 쓰기가 가능하며 이 배열 자체에 결과를 덮어씌웁니다(In-place update).
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

/**
 * @function main
 * @brief 그레이디언트 값에 학습률을 곱한 뒤 기존 파라미터에서 빼는 방식(param = param - lr * grad)으로 값을 갱신합니다. (What)
 * GPU 코어들을 병렬로 사용하여 수많은 모델 파라미터를 한 번에 업데이트하기 위해(Why) 실행되는 메인 셰이더입니다.
 * @param global_id 워크그룹 내 스레드의 3차원 전역 인덱스 변수입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 1차원 전역 인덱스를 가져옵니다. 각 스레드가 하나의 파라미터를 담당합니다. (How)
  let idx = global_id.x;
  
  // 현재 인덱스가 처리해야 할 요소 개수(numElements)를 넘어갔는지 검사합니다. (What)
  // 배열 크기 이상의 메모리 접근을 방지하기 위함입니다. (Why)
  if (idx >= params.numElements) {
    return;
  }
  
  // 현재 파라미터 값에서 (학습률 * 기울기) 만큼을 차감하여 새로운 값으로 갱신(덮어쓰기)합니다. (How)
  // 전형적인 옵티마이저(SGD)의 스텝 연산입니다. (What)
  param[idx] = param[idx] - params.lr * grad[idx];
}
`;
