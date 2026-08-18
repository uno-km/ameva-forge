/**
 * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
 * 수정 이력:
 * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * - 2026-08-18 00:30:00 +0900: Fix(SCRUM-154/VULN-02): Generic 3-parameter reduction for 3D/4D tensors
 */
export const SUM_AXIS_WGSL = `
/**
 * 이 구조체(Params)는 임의 축에 대한 텐서 축소(Sum Along Axis) 연산에 필요한 메타데이터를 담고 있습니다.
 * 3차원 이상의 고차원 텐서에서도 일반화된 (outer_size, reduction_size, inner_stride) 3-파라미터 체계를 지원합니다.
 */
struct Params {
  outer_size: u32,     // 축소 축 이전의 외부 배치/차원들의 곱
  reduction_size: u32, // 축소할 대상 축의 원소 개수 (Reduction Dimension Size)
  inner_stride: u32,   // 축소 축 이후의 내부 차원들의 스트라이드 곱
  output_numel: u32,   // 결과 텐서의 총 원소 개수 (outer_size * inner_stride)
  workgroups_x: u32,   // 2D 디스패치 분할을 위한 X축 워크그룹 수
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 축소 메타데이터 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 축소 연산을 수행할 원본 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 축소된 결과가 저장될 출력 텐서입니다.

/**
 * main 함수는 출력 텐서의 각 원소에 대해 입력 텐서의 reduction_size개 원소들을 순회하며 합산합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let workgroups_x = params.workgroups_x;
  let out_idx = global_id.x + global_id.y * workgroups_x * 64u;

  if (out_idx >= params.output_numel) {
    return;
  }

  let inner_stride = params.inner_stride;
  let reduction_size = params.reduction_size;
  let outer_idx = out_idx / inner_stride;
  let inner_idx = out_idx % inner_stride;
  let slice_stride = reduction_size * inner_stride;
  let base_offset = outer_idx * slice_stride + inner_idx;

  var sum = 0.0;
  for (var r = 0u; r < reduction_size; r = r + 1u) {
    sum += input[base_offset + r * inner_stride];
  }
  output[out_idx] = sum;
}
`;
