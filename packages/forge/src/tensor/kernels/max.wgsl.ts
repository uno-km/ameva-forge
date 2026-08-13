/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
export const MAX_WGSL = `
/**
 * 이 구조체(Params)는 텐서의 최댓값을 구하는 Reduction(리덕션) 연산에 필요한 정보를 담고 있습니다.
 * 요소의 전체 개수를 전달하여 버퍼 경계를 넘는 접근을 방지하기 위해 존재합니다.
 */
struct Params {
  numElements: u32, // 최댓값을 찾을 전체 배열 원소의 개수입니다.
  pad1: u32, // WebGPU의 16바이트 정렬을 맞추기 위한 빈 패딩 변수입니다.
  pad2: u32, // 16바이트 정렬을 위한 두 번째 패딩 변수입니다.
  pad3: u32, // 16바이트 정렬을 위한 세 번째 패딩 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 메타데이터 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 최댓값을 탐색할 원본 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 워크그룹별 부분 최댓값이 저장될 출력 텐서입니다.

// 하나의 워크그룹(256개의 스레드) 내에서 데이터를 공유하고 리덕션을 수행하기 위해 존재하는 공유 메모리 공간입니다.
var<workgroup> shared: array<f32, 256>;

/**
 * main 함수는 트리 기반의 리덕션(Tree-based Reduction) 알고리즘을 사용하여 배열 내 원소들의 최댓값을 계산합니다.
 * 방대한 데이터를 병렬로 빠르게 비교압축하기 위해 공유 메모리(shared)와 배리어(barrier) 동기화를 사용합니다.
 */
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let gid = global_id.x; // 글로벌 단위에서 현재 스레드의 1차원 인덱스입니다.
  let lid = local_id.x; // 워크그룹 내부에서 현재 스레드의 1차원 인덱스(0~255)입니다.
  let wid = workgroup_id.x; // 현재 스레드가 속한 워크그룹의 ID(인덱스)입니다.
  
  // 글로벌 인덱스가 데이터 크기 이내라면 입력 데이터를, 벗어난다면 부동소수점의 최소값(-FLT_MAX)을 공유 메모리에 로드합니다.
  if (gid < params.numElements) {
    shared[lid] = input[gid];
  } else {
    shared[lid] = -3.402823e+38; // 쓰레기값을 방지하기 위한 최소값 초기화입니다.
  }
  
  // 공유 메모리 로드가 완전히 끝날 때까지 워크그룹 내의 모든 스레드를 대기시킵니다.
  workgroupBarrier();
  
  // 트리 기반 병렬 리덕션 루프입니다.
  // 활성화된 스레드 수를 절반씩 줄여가면서(128 -> 64 -> ... -> 1) 두 요소씩 비교해 최댓값을 찾습니다.
  for (var s = 128u; s > 0u; s >>= 1u) {
    // 현재 단계에서 값을 비교하고 갱신할 권한이 있는 스레드만 실행합니다.
    if (lid < s) {
      shared[lid] = max(shared[lid], shared[lid + s]); // 자신의 값과 s만큼 떨어진 옆의 값을 비교해 큰 값을 저장합니다.
    }
    // 데이터 경합(Data Race)을 막고 다음 단계를 안전하게 수행하기 위해 스레드 동기화를 수행합니다.
    workgroupBarrier();
  }
  
  // 리덕션이 완료되면 공유 메모리의 0번 인덱스에 현재 워크그룹의 전체 최댓값이 남게 됩니다.
  // 0번 스레드가 이를 대표로 전역 출력 버퍼에 기록합니다.
  if (lid == 0u) {
    output[wid] = shared[0];
  }
}
`;
