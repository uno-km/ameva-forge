/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-326 2D Nearest & Bilinear Upsampling WGSL Compute Kernel
 *
 * WHAT: 디퓨전(Stable Diffusion UNet 업샘플링 블록 및 VAE 디코더)의 핵심 공간 해상도 2배 확대 연산 WGSL 커널입니다.
 * WHY: 잠재 공간(Latent: 64x64)에서 고해상도 픽셀 맵(512x512)으로의 점진적 복원을 고속 WebGPU 병렬 연산으로 처리하기 위해 존재합니다.
 * HOW: Nearest Neighbor(모드 0)와 Bilinear Interpolation(모드 1)을 단일 컴퓨트 파이프라인에서 지원하며,
 *      16바이트 정렬 규격(8 x 4바이트 = 32바이트)을 100% 준수합니다.
 */

export const UPSAMPLE2D_WGSL = `
struct Params {
  N: u32,             // 배치 크기
  C: u32,             // 채널 개수
  H_in: u32,          // 입력 특징 맵 높이
  W_in: u32,          // 입력 특징 맵 너비
  H_out: u32,         // 출력 특징 맵 높이
  W_out: u32,         // 출력 특징 맵 너비
  mode: u32,          // 0: Nearest, 1: Bilinear
  workgroups_x: u32,  // 2D 디스패치 선형 복원용 X 크기
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;
  let total_elements = params.N * params.C * params.H_out * params.W_out;

  if (idx >= total_elements) {
    return;
  }

  // 1D 인덱스를 N, C, H_out, W_out 좌표로 역산
  let w_out = idx % params.W_out;
  var rem = idx / params.W_out;
  let h_out = rem % params.H_out;
  rem = rem / params.H_out;
  let c = rem % params.C;
  let n = rem / params.C;

  let base_in_offset = (n * params.C + c) * (params.H_in * params.W_in);

  if (params.mode == 0u) {
    // 1. Nearest Neighbor Mode
    let scale_h = f32(params.H_out) / f32(params.H_in);
    let scale_w = f32(params.W_out) / f32(params.W_in);

    let h_in = min(u32(floor(f32(h_out) / scale_h)), params.H_in - 1u);
    let w_in = min(u32(floor(f32(w_out) / scale_w)), params.W_in - 1u);

    let in_idx = base_in_offset + h_in * params.W_in + w_in;
    output[idx] = input[in_idx];
  } else {
    // 2. Bilinear Interpolation Mode (align_corners = false)
    let scale_h = f32(params.H_in) / f32(params.H_out);
    let scale_w = f32(params.W_in) / f32(params.W_out);

    let real_h = (f32(h_out) + 0.5) * scale_h - 0.5;
    let real_w = (f32(w_out) + 0.5) * scale_w - 0.5;

    let h0 = u32(max(0.0, floor(real_h)));
    let w0 = u32(max(0.0, floor(real_w)));
    let h1 = min(h0 + 1u, params.H_in - 1u);
    let w1 = min(w0 + 1u, params.W_in - 1u);

    let dh = clamp(real_h - f32(h0), 0.0, 1.0);
    let dw = clamp(real_w - f32(w0), 0.0, 1.0);

    let v00 = input[base_in_offset + h0 * params.W_in + w0];
    let v01 = input[base_in_offset + h0 * params.W_in + w1];
    let v10 = input[base_in_offset + h1 * params.W_in + w0];
    let v11 = input[base_in_offset + h1 * params.W_in + w1];

    let top = v00 * (1.0 - dw) + v01 * dw;
    let bottom = v10 * (1.0 - dw) + v11 * dw;
    output[idx] = top * (1.0 - dh) + bottom * dh;
  }
}
`;
