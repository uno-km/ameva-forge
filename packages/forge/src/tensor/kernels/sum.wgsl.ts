/**
 * 파일 생성: 2026-08-12 12:14:52
 * 수정 내역:
 * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 */
export const SUM_WGSL = `
// 구조체: Params
// 목적: 합계(Sum) 연산에 필요한 메타데이터와 패딩을 정의합니다.
// 작동 방식: 배열의 전체 크기를 제공하며, 16바이트 정렬을 준수합니다.
struct Params {
  // 변수: numElements
  // 목적: 더해야 할 입력 배열의 전체 원소 개수를 나타냅니다.
  // 작동 방식: 전역 인덱스가 유효 범위를 벗어나는지 검사하는 용도로 사용됩니다.
  numElements: u32,
  // 변수: pad1
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트 배수로 맞춰 메모리 접근 성능을 높입니다.
  pad1: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트 배수로 맞춰 메모리 접근 성능을 높입니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트 배수로 맞춰 메모리 접근 성능을 높입니다.
  pad3: u32,
};

// 변수: params
// 목적: 유니폼 버퍼를 통해 워크그룹 외부에서 메타데이터를 주입받습니다.
// 작동 방식: 바인딩 0에 할당되어 전체 요소 개수를 모든 스레드에 제공합니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 합계를 구할 대상이 되는 데이터를 담은 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 1에 할당되며, 각 스레드가 자신의 위치에 해당하는 값을 읽어옵니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 목적: 각 워크그룹 내에서의 부분 합계(Partial sum)를 저장할 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되며, 최종적으로 워크그룹 개수만큼의 결과가 저장됩니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 변수: shared
// 목적: 워크그룹 내 스레드들이 공유하는 로컬 메모리(Shared memory)입니다.
// 작동 방식: 256 크기의 배열로 할당되어 빠른 Reduction(축소) 연산을 위한 캐시 역할을 합니다.
var<workgroup> shared: array<f32, 256>;

// 함수: main
// 목적: 배열 요소들의 총합을 구하기 위한 병렬 Reduction(축소) 알고리즘을 수행합니다.
// 작동 방식: 워크그룹 내에서 공유 메모리를 사용하여 트리(Tree) 구조로 단계별 덧셈을 수행합니다.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  // 변수: gid
  // 목적: 전체 스레드 중 현재 스레드의 고유 1차원 전역 인덱스입니다.
  // 작동 방식: global_id.x 값을 가져와 입력 데이터 접근에 사용합니다.
  let gid = global_id.x;

  // 변수: lid
  // 목적: 현재 워크그룹 내에서의 로컬 인덱스(0~255)입니다.
  // 작동 방식: local_id.x 값을 가져와 공유 메모리 접근 및 축소 연산 인덱스로 사용합니다.
  let lid = local_id.x;

  // 변수: wid
  // 목적: 현재 속해 있는 워크그룹의 고유 ID입니다.
  // 작동 방식: workgroup_id.x 값을 가져와 부분 합 결과를 저장할 위치를 결정합니다.
  let wid = workgroup_id.x;
  
  // 제어문: if-else
  // 목적: 입력 데이터를 로컬 공유 메모리에 복사하면서, 범위를 벗어난 공간을 0으로 초기화합니다.
  // 작동 방식: gid가 유효한 원소 범위 안에 있으면 input[gid]를, 벗어나면 0.0을 shared에 할당합니다.
  if (gid < params.numElements) {
    shared[lid] = input[gid];
  } else {
    shared[lid] = 0.0;
  }
  
  // 동기화: workgroupBarrier()
  // 목적: 워크그룹 내의 모든 스레드가 공유 메모리에 데이터를 쓸 때까지 대기합니다.
  // 작동 방식: 초기 데이터 적재(Load)가 완전히 끝난 뒤에만 다음 Reduction 단계를 진행하도록 보장합니다.
  workgroupBarrier();
  
  // 반복문: for
  // 목적: 워크그룹 내 256개의 요소를 트리 기반 병렬 Reduction 방식으로 더합니다.
  // 작동 방식: s 변수를 128부터 시작하여 0보다 클 때까지 절반으로 줄여가며 (비트 시프트 연산) 부분 합을 구합니다.
  for (var s = 128u; s > 0u; s >>= 1u) {
    // 제어문: if
    // 목적: 유효한 절반의 스레드들만 덧셈 연산에 참여하도록 제한합니다.
    // 작동 방식: 로컬 인덱스가 현재 단계의 s보다 작을 경우에만 shared[lid]와 shared[lid + s]를 더합니다.
    if (lid < s) {
      shared[lid] += shared[lid + s];
    }
    // 동기화: workgroupBarrier()
    // 목적: 각 단계의 덧셈이 모든 참여 스레드에서 끝날 때까지 대기합니다.
    // 작동 방식: 이전 단계의 덧셈 결과가 완전히 기록된 후 다음 단계를 진행하도록 합니다.
    workgroupBarrier();
  }
  
  // 제어문: if
  // 목적: Reduction 연산이 완료된 최종 합계를 전역 메모리에 기록합니다.
  // 작동 방식: 로컬 인덱스가 0인 첫 번째 스레드가 워크그룹의 최종 합(shared[0])을 output 버퍼의 wid 위치에 저장합니다.
  if (lid == 0u) {
    output[wid] = shared[0];
  }
}
`;
