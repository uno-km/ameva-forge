/**
 * 파일 생성: 2026-08-12 12:23:09
 * 수정 내역:
 * - 2026-08-12 12:23:09: Docs: Build Apache-style docs and unify tests (fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
 */
export const WHERE_WGSL = `
// 구조체: Params
// 목적: 조건부 분기(Where) 연산에 사용되는 메타데이터를 저장합니다.
// 작동 방식: 처리할 전체 요소 크기(size)와 패딩 값들을 통해 16바이트 정렬된 메모리 구조를 형성합니다.
struct Params {
  // 변수: size
  // 목적: 배열의 전체 요소 수를 지정합니다.
  // 작동 방식: 커널에서 각 스레드가 유효한 범위 내에 있는지 확인하는 데 사용됩니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향 워크그룹의 총 개수입니다.
  // 작동 방식: 글로벌 인덱스 변환 시 X축 길이를 곱하는 계수로 사용됩니다.
  workgroups_x: u32,
  // 변수: pad2 ~ pad7
  // 목적: 메모리 정렬(Alignment)을 맞추기 위한 여유 공간(패딩)들입니다.
  // 작동 방식: WGSL 유니폼 버퍼의 레이아웃 규칙에 맞추기 위해 사용됩니다.
  pad2: u32,
  pad3: u32,
  pad4: u32,
  pad5: u32,
  pad6: u32,
  pad7: u32,
};

// 변수: params
// 목적: 외부에서 제공되는 파라미터 구조체를 바인딩합니다.
// 작동 방식: 바인딩 0에 읽기 전용으로 매핑됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: cond
// 목적: 요소별로 어느 값을 선택할지 결정하는 조건(Condition) 배열입니다.
// 작동 방식: 값이 0보다 크면 참(True), 그렇지 않으면 거짓(False)으로 평가됩니다.
@group(0) @binding(1) var<storage, read> cond: array<f32>;

// 변수: x
// 목적: 조건이 참(True)일 때 선택될 데이터 배열입니다.
// 작동 방식: 바인딩 2에 할당됩니다.
@group(0) @binding(2) var<storage, read> x: array<f32>;

// 변수: y
// 목적: 조건이 거짓(False)일 때 선택될 데이터 배열입니다.
// 작동 방식: 바인딩 3에 할당됩니다.
@group(0) @binding(3) var<storage, read> y: array<f32>;

// 변수: out
// 목적: 조건에 따라 x 또는 y에서 선택된 결과가 저장될 배열입니다.
// 작동 방식: 바인딩 4에 할당되어 계산 결과를 기록합니다.
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: cond 배열의 값에 따라 병렬로 x 또는 y의 요소를 선택하여 out 배열에 씁니다 (TensorFlow/PyTorch의 where 함수와 유사).
// 작동 방식: 각 스레드가 조건 평가를 거쳐 선택한 값을 기록합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산할 전체 배열 요소의 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 스레드가 담당하는 배열 요소의 1차원 인덱스입니다.
  // 작동 방식: 2차원 워크그룹 배열 인덱스를 1차원으로 평면화하여 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 배열의 유효 범위를 넘어가는 스레드가 메모리에 접근하지 않게 합니다.
  // 작동 방식: 인덱스가 전체 크기 이상이면 함수를 끝냅니다.
  if (idx >= num_elements) {
    return;
  }

  // 제어문: if-else
  // 목적: 조건에 맞게 x 배열과 y 배열 중 하나의 값을 선택합니다.
  // 작동 방식: cond[idx]가 0보다 크면 x[idx]를, 아니면 y[idx]를 out[idx]에 할당합니다.
  if (cond[idx] > 0.0) {
    out[idx] = x[idx];
  } else {
    out[idx] = y[idx];
  }
}
`;
