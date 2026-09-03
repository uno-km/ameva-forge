/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-310 GroupNorm (32 groups) & Fused SiLU WGSL Compute Kernel
 *
 * WHAT: 디퓨전(Stable Diffusion UNet 및 VAE)의 핵심 정규화인 Group Normalization 및 SiLU 융합 WGSL 커널입니다.
 * WHY: UNet 전체에 수십 번 적용되는 GroupNorm(32그룹)을 2-Pass 병렬 트리 축소(Tree Reduction)와 아핀(Affine) 변환으로 처리하기 위해 존재합니다.
 * HOW: Pass 1에서 (배치, 그룹)별 평균과 분산을 워크그룹 공유 메모리로 고속 계산하고,
 *      Pass 2에서 정규화(x_norm = (x - mean) / sqrt(var + eps)), gamma/beta 아핀 변환 및 선택적 Fused SiLU를 적용합니다.
 *      16바이트 정렬 규격(12 x 4바이트 = 48바이트)을 100% 준수합니다.
 */

export const GROUP_NORM_STATS_WGSL = `
struct Params {
  N: u32,
  C: u32,
  H: u32,
  W: u32,
  num_groups: u32,
  channels_per_group: u32,
  fuse_silu: u32,
  workgroups_x: u32,
  eps: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<vec2<f32>>; // (mean, var) per (n, g)

var<workgroup> wg_sum: array<f32, 64>;
var<workgroup> wg_sq_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let ng_idx = workgroup_id.x; // (n * num_groups + g)
  let total_groups = params.N * params.num_groups;
  if (ng_idx >= total_groups) {
    return;
  }

  let g = ng_idx % params.num_groups;
  let n = ng_idx / params.num_groups;

  let group_size = params.channels_per_group * params.H * params.W;
  let hw = params.H * params.W;
  let base_c = g * params.channels_per_group;

  var local_sum: f32 = 0.0;
  var local_sq_sum: f32 = 0.0;

  // Stride over all elements in this (n, g) group
  var i = local_id.x;
  while (i < group_size) {
    let c_offset = i / hw;
    let hw_offset = i % hw;
    let actual_c = base_c + c_offset;
    let in_idx = (n * params.C + actual_c) * hw + hw_offset;
    let val = input[in_idx];
    local_sum = local_sum + val;
    local_sq_sum = local_sq_sum + val * val;
    i = i + 64u;
  }

  wg_sum[local_id.x] = local_sum;
  wg_sq_sum[local_id.x] = local_sq_sum;
  workgroupBarrier();

  // Parallel reduction in workgroup shared memory
  var stride = 32u;
  while (stride > 0u) {
    if (local_id.x < stride) {
      wg_sum[local_id.x] = wg_sum[local_id.x] + wg_sum[local_id.x + stride];
      wg_sq_sum[local_id.x] = wg_sq_sum[local_id.x] + wg_sq_sum[local_id.x + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (local_id.x == 0u) {
    let mean = wg_sum[0] / f32(group_size);
    let variance = max(0.0, (wg_sq_sum[0] / f32(group_size)) - (mean * mean));
    stats[ng_idx] = vec2<f32>(mean, variance);
  }
}
`;

export const GROUP_NORM_APPLY_WGSL = `
struct Params {
  N: u32,
  C: u32,
  H: u32,
  W: u32,
  num_groups: u32,
  channels_per_group: u32,
  fuse_silu: u32,
  workgroups_x: u32,
  eps: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<vec2<f32>>;  // (mean, var)
@group(0) @binding(3) var<storage, read> gamma: array<f32>;        // scale [C]
@group(0) @binding(4) var<storage, read> beta: array<f32>;         // bias [C]
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn stable_silu(x: f32) -> f32 {
  let clamped_x = clamp(x, -88.0, 88.0);
  let sig = 1.0 / (1.0 + exp(-clamped_x));
  return x * sig;
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;
  let total_elements = params.N * params.C * params.H * params.W;

  if (idx >= total_elements) {
    return;
  }

  let hw = params.H * params.W;
  let c = (idx / hw) % params.C;
  let n = idx / (params.C * hw);
  let g = c / params.channels_per_group;

  let ng_idx = n * params.num_groups + g;
  let stat = stats[ng_idx];
  let mean = stat.x;
  let variance = stat.y;

  let inv_std = inverseSqrt(variance + params.eps);
  let x_norm = (input[idx] - mean) * inv_std;

  var y = x_norm * gamma[c] + beta[c];

  if (params.fuse_silu == 1u) {
    y = stable_silu(y);
  }

  output[idx] = y;
}
`;
