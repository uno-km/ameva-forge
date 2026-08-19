/**
 * 파일 생성일: 2026-08-18T12:05:00+09:00
 * 역할: 축 방향 최댓값 리덕션 (Max Reduction Along Axis) WGSL 커널
 * 목적: Softmax의 수치적 안정성(x - max(x)) 및 축별 Max 연산을 GPU에서 고속 병렬 처리하기 위함
 */
export const MAX_AXIS_WGSL = `
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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let workgroups_x = params.workgroups_x;
  let out_idx = global_id.x + global_id.y * workgroups_x * 64u;

  if (out_idx >= params.output_numel) {
    return;
  }

  let inner_stride = max(params.inner_stride, 1u);
  let reduction_size = params.reduction_size;
  let outer_idx = out_idx / inner_stride;
  let inner_idx = out_idx % inner_stride;
  let slice_stride = reduction_size * inner_stride;
  let base_offset = outer_idx * slice_stride + inner_idx;

  var max_val = -3.402823e+38;
  for (var r = 0u; r < reduction_size; r = r + 1u) {
    let val = input[base_offset + r * inner_stride];
    if (val > max_val || val != val) {
      max_val = val;
      if (val != val) {
        break;
      }
    }
  }
  output[out_idx] = max_val;
}
`;
