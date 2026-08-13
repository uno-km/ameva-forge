/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const DIV_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서 간의 요소별 나눗셈(division) 연산을 수행할 때 필요한 파라미터입니다. (What)
 * 셰이더 내에서 배열 크기 경계를 확인하고 3D 스레드 인덱스를 1차원으로 풀기 위해 사용됩니다. (Why)
 */
struct Params {
  // 전체 요소(element)의 개수입니다. 배열 인덱스 초과를 막기 위해 사용됩니다.
  size: u32,
  // X 차원의 워크그룹 총 개수입니다. 2차원(Y방향) 인덱스 계산을 위해 사용됩니다.
  workgroups_x: u32,
  // 16바이트(float4) 메모리 정렬을 위한 여유 패딩 변수입니다.
  pad2: u32,
  // 16바이트 메모리 정렬을 위한 여유 패딩 변수입니다.
  pad3: u32,
};

// params: GPU와 CPU 간 데이터 통신을 위한 Uniform 버퍼입니다. 텐서 크기 정보를 전달합니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 분자(나누어지는 수, dividend) 역할을 하는 첫 번째 입력 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 분모(나누는 수, divisor) 역할을 하는 두 번째 입력 배열입니다 (읽기 전용).
@group(0) @binding(2) var<storage, read> b: array<f32>;
// out: 나눗셈 연산의 몫이 저장되는 출력 배열입니다 (읽기/쓰기 가능).
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

/**
 * @function main
 * @brief 스레드 인덱스에 따라 a 텐서의 값을 b 텐서의 값으로 나누어 그 결과를 out 텐서에 기록합니다. (What)
 * 요소 단위(Element-wise)의 병렬 나눗셈을 통해 대규모 데이터의 정규화 등의 처리를 매우 빠르게 수행하기 위함입니다. (Why)
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스입니다. (How)
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 연산을 수행할 전체 원소의 수를 가져옵니다.
  let num_elements = params.size;
  // X축으로 할당된 워크그룹의 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 글로벌 스레드 ID의 X, Y 좌표와 워크그룹 수, 워크그룹 사이즈(64)를 곱하고 더하여 1차원 선형 인덱스(idx)를 구합니다. (How)
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 계산된 인덱스가 전체 요소 수(num_elements)보다 작은지 검사합니다. (What)
  // 텐서의 크기보다 초과된 메모리 영역에 잘못 접근하는 오류를 방지하기 위해 사용됩니다. (Why)
  if (idx < num_elements) {
    // 해당 위치의 a 값을 b 값으로 나누어 결과 배열(out)에 저장합니다. (How)
    // (주의: WGSL에서 부동소수점 0으로 나누기 발생 시 무한대(Infinity)나 NaN이 들어갈 수 있습니다.)
    out[idx] = a[idx] / b[idx];
  }
}
`;
