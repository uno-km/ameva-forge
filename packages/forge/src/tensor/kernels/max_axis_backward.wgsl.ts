/**
 * 파일 생성일: 2026-08-18T13:20:00+09:00
 * 역할: 축 방향 최댓값 역전파 (Max Reduction Backward Along Axis) WGSL 커널
 * 목적: GPU 상에서 x.max(axis).backward() 호출 시 기울기 전파 및 중복 최댓값 분산 처리
 */
export const MAX_AXIS_BACKWARD_WGSL = `
struct Params {
  outer_size: u32,
  reduction_size: u32,
  inner_stride: u32,
  input_numel: u32,
  workgroups_x: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> grad_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x + gid.y * params.workgroups_x * 64u;
  if (linear >= params.input_numel) {
    return;
  }

  let inner_stride = max(params.inner_stride, 1u);
  let reduction_size = max(params.reduction_size, 1u);
  let inner = linear % inner_stride;
  let tmp = linear / inner_stride;
  let r = tmp % reduction_size;
  let outer = tmp / reduction_size;

  let reduced_idx = outer * inner_stride + inner;

  var max_val = -3.402823e+38;
  for (var j: u32 = 0u; j < reduction_size; j = j + 1u) {
    let idx = outer * reduction_size * inner_stride + j * inner_stride + inner;
    max_val = max(max_val, x[idx]);
  }

  var count: f32 = 0.0;
  for (var j: u32 = 0u; j < params.reduction_size; j = j + 1u) {
    let idx = outer * params.reduction_size * params.inner_stride + j * params.inner_stride + inner;
    if (x[idx] == max_val) {
      count = count + 1.0;
    }
  }

  if (x[linear] == max_val && count > 0.0) {
    grad_x[linear] = grad_out[reduced_idx] / count;
  } else {
    grad_x[linear] = 0.0;
  }
}
`;
