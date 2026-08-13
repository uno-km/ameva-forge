/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const SUB_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터(Uniform parameters)들을 정의합니다.
// 작동 방식: 배열의 크기(size)와 2차원 워크그룹 배열의 X축 크기(workgroups_x)를 제공합니다.
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
  // 목적: 메모리 16바이트 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 메모리 16바이트 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad3: u32,
};

// 변수: params
// 목적: 셰이더 실행에 필요한 설정값을 담고 있는 유니폼 버퍼 변수입니다.
// 작동 방식: 바인딩 0에 할당되어 모든 워크그룹이 공유하여 읽습니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: a
// 목적: 뺄셈 연산의 피연산자 A(minuend)를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 1에 할당되어 첫 번째 입력 텐서 데이터를 제공합니다.
@group(0) @binding(1) var<storage, read> a: array<f32>;

// 변수: b
// 목적: 뺄셈 연산의 피연산자 B(subtrahend)를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되어 두 번째 입력 텐서 데이터를 제공합니다.
@group(0) @binding(2) var<storage, read> b: array<f32>;

// 변수: out
// 목적: 뺄셈 연산 결과(A - B)를 저장할 읽기/쓰기 가능한 출력 버퍼입니다.
// 작동 방식: 바인딩 3에 할당되며, 병렬 처리된 결과가 각 인덱스 위치에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: 두 텐서의 요소별(element-wise) 뺄셈 연산을 병렬로 수행하는 커널 진입점입니다.
// 작동 방식: 각 스레드가 자신의 인덱스(idx)를 계산한 뒤 'a[idx] - b[idx]' 연산을 수행하고 out 배열에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산할 전체 원소 개수를 로컬에 할당합니다.
  // 작동 방식: 유니폼 구조체(params)에서 size 필드를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수를 로컬에 할당합니다.
  // 작동 방식: 유니폼 구조체(params)에서 workgroups_x 필드를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 스레드가 처리할 데이터의 1차원 인덱스를 계산합니다.
  // 작동 방식: 현재 워크그룹의 (y * 워크그룹X개수 * 64) 오프셋에 x 인덱스를 더해 글로벌 인덱스를 평면화합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 배열 크기를 넘어서는 유효하지 않은 인덱스에 접근하는 것을 방지합니다.
  // 작동 방식: idx가 num_elements보다 작을 때만 아래 연산을 수행합니다.
  if (idx < num_elements) {
    // 연산: out[idx] 기록
    // 목적: 요소별 뺄셈 결과를 저장합니다.
    // 작동 방식: 인덱스에 해당하는 a 값에서 b 값을 뺀 후 out 배열의 같은 위치에 씁니다.
    out[idx] = a[idx] - b[idx];
  }
}
`;
