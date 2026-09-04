/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: High-Performance WebGPU STT Short-Time Fourier Transform (STFT) Kernel
 *
 * WHAT: 오디오 16kHz PCM 파형으로부터 각 프레임별 Hanning Window 및 복소수 DFT(이산 푸리에 변환)를
 *      GPU의 수천 개 워크그룹 스레드에서 병렬 계산하여 매그니튜드(Magnitude) 버퍼를 VRAM 내에서 생성하는 WGSL 컴퓨트 커널입니다.
 * WHY: 1분 오디오 기준 4.8억 번의 CPU 삼각함수 연산 병목을 제거하고, 순수 WebGPU 병렬 연산으로 수십 밀리초 내에 처리하기 위함입니다.
 * HOW: Frame 및 Bin(k) 인덱스를 2D 그리드로 분할하여, 스레드당 400개 샘플의 Hanning 가중 삼각함수를 곱셈 누산(FMA)합니다.
 */

export const STT_STFT_WGSL = `
struct STFTParams {
  num_frames: u32,
  n_fft: u32,       // 400
  hop_length: u32,  // 160
  n_bins: u32,      // 201
  workgroups_x: u32,
  pcm_length: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> params: STFTParams;
@group(0) @binding(1) var<storage, read> pcm_samples: array<f32>;
@group(0) @binding(2) var<storage, read_write> stft_magnitudes: array<f32>; // [num_frames, n_bins]

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let total_entries = params.num_frames * params.n_bins;
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;
  if (idx >= total_entries) {
    return;
  }

  let frame = idx / params.n_bins;
  let k = idx % params.n_bins;
  let start = frame * params.hop_length;

  var real = 0.0;
  var imag = 0.0;
  let two_pi_over_nfft = 6.28318530717958647692 / f32(params.n_fft);
  let k_f32 = f32(k);

  for (var n = 0u; n < params.n_fft; n = n + 1u) {
    let sample_idx = start + n;
    var sample = 0.0;
    if (sample_idx < params.pcm_length) {
      sample = pcm_samples[sample_idx];
    }
    let n_f32 = f32(n);
    // Hanning Window: w[n] = 0.5 * (1.0 - cos(2 * pi * n / n_fft))
    let win = 0.5 * (1.0 - cos(two_pi_over_nfft * n_f32));
    let sample_win = sample * win;

    let angle = -two_pi_over_nfft * k_f32 * n_f32;
    real = real + sample_win * cos(angle);
    imag = imag + sample_win * sin(angle);
  }

  stft_magnitudes[idx] = sqrt(real * real + imag * imag);
}
`;
