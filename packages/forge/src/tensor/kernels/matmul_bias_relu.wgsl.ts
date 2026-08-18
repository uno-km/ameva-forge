/**
 * 파일 생성일: 2026-08-12
 * 수정일: 2026-08-18 (Release 2.0 SCRUM-203 고도화)
 *
 * WHAT: 16x16 워크그룹 공유 메모리(Shared Memory) 기반 Fused GEMM (MatMul + Bias + ReLU/GELU) 커널입니다.
 * WHY: Linear Layer 및 FFN 계층에서 중간 버퍼 VRAM 할당과 메모리 왕복 대역폭 소모를 100% 제거하기 위해 존재합니다.
 * HOW: 공유 메모리 타일링으로 A, B 행렬곱을 수행한 후, 레지스터 레벨에서 Bias 덧셈과 활성화 함수(ReLU/GELU)를 단일 패스로 처리합니다.
 */

export const MATMUL_BIAS_RELU_WGSL = `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  offsetY: u32,
  has_bias: u32,  // 1: bias 적용, 0: 생략
  activation_type: u32, // 0: None, 1: ReLU, 2: GELU
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

fn compute_gelu(x: f32) -> f32 {
  let sqrt_2_over_pi = 0.7978845608;
  let coef = 0.044715;
  let inner = sqrt_2_over_pi * (x + coef * x * x * x);
  return 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let local_row = local_id.y;
  let local_col = local_id.x;

  let global_row_c = workgroup_id.y * 16u + local_row + params.offsetY;
  let global_col_c = (workgroup_id.x + workgroup_id.z * 65535u) * 16u + local_col;

  let num_tiles = (params.K + 15u) / 16u;
  var acc: f32 = 0.0;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let global_row_a = global_row_c;
    let global_col_a = t * 16u + local_col;

    if (global_row_a < params.M && global_col_a < params.K) {
      tileA[local_row][local_col] = a[global_row_a * params.K + global_col_a];
    } else {
      tileA[local_row][local_col] = 0.0;
    }

    let global_row_b = t * 16u + local_row;
    let global_col_b = global_col_c;

    if (global_row_b < params.K && global_col_b < params.N) {
      tileB[local_row][local_col] = b[global_row_b * params.N + global_col_b];
    } else {
      tileB[local_row][local_col] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      acc = acc + tileA[local_row][k] * tileB[k][local_col];
    }

    workgroupBarrier();
  }

  if (global_row_c < params.M && global_col_c < params.N) {
    if (params.has_bias == 1u) {
      acc = acc + bias[global_col_c];
    }

    if (params.activation_type == 1u) {
      // ReLU
      acc = max(acc, 0.0);
    } else if (params.activation_type == 2u) {
      // GELU
      acc = compute_gelu(acc);
    }

    c[global_row_c * params.N + global_col_c] = acc;
  }
}
`;
