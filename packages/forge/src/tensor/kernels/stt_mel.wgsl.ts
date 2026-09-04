/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-335 WebGPU STT Log Mel-Filterbank Compute Kernel
 *
 * WHAT: 오디오 STFT 프레임 에너지를 80개 삼각 멜-필터뱅크(Mel-Filterbank)에 투영하는 WGSL 컴퓨트 셰이더입니다.
 * WHY: 오디오 음향 특징 추출(Mel-Spectrogram)을 WebGPU 하드웨어에서 초고속 병렬 디스패치하기 위함입니다.
 */

export const STT_MEL_WGSL = `
struct Params {
  num_frames: u32,
  num_mels: u32,
  n_fft_bins: u32,
  workgroups_x: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stft_magnitudes: array<f32>; // [num_frames, n_fft_bins]
@group(0) @binding(2) var<storage, read> mel_filterbank: array<f32>;   // [num_mels, n_fft_bins]
@group(0) @binding(3) var<storage, read_write> output_mels: array<f32>; // [num_frames, num_mels]

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let total_entries = params.num_frames * params.num_mels;
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;
  if (idx >= total_entries) {
    return;
  }

  let frame = idx / params.num_mels;
  let mel = idx % params.num_mels;

  var energy = 0.0;
  let frame_offset = frame * params.n_fft_bins;
  let mel_offset = mel * params.n_fft_bins;

  for (var k = 0u; k < params.n_fft_bins; k = k + 1u) {
    let mag = stft_magnitudes[frame_offset + k];
    let weight = mel_filterbank[mel_offset + k];
    energy = energy + mag * weight;
  }

  // Log compression: log10(max(energy, 1e-5))
  let log_val = log(max(energy, 0.00001)) * 0.4342944819; // 1 / ln(10)
  output_mels[idx] = log_val;
}
`;
