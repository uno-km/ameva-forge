/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-335 WebGPU TTS Rosenberg Glottal Flow & Resonator Synthesis Kernel
 *
 * WHAT: Rosenberg 성문 펄스(Glottal Pulse)와 모음 포먼트 공진을 WebGPU 워크그룹에서 병렬 계산하는 WGSL 컴퓨트 셰이더입니다.
 * WHY: CPU 단일 스레드 음성 합성 루프를 완전히 탈피하여 GPU 수천 코어에서 동시 병렬 파형을 초고속으로 합성하기 위함입니다.
 * HOW: g(t) = 3t^2 - 2t^3 성문 펄스를 시간 축 병렬 디스패치하고, 포먼트 필터링을 VRAM 내에서 일괄 수행합니다.
 */

export const TTS_SYNTH_WGSL = `
struct Params {
  total_samples: u32,
  sample_rate: f32,
  f0: f32,
  workgroups_x: u32,
  vowel_idx: u32,
  f1: f32,
  f2: f32,
  f3: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output_pcm: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroups_x) * 64u + local_id.x;
  if (idx >= params.total_samples) {
    return;
  }

  let sr = params.sample_rate;
  let f0 = params.f0;
  let t_period = sr / f0;

  let sample_in_period = f32(idx) % t_period;
  let phase = sample_in_period / t_period;

  var glottal = 0.0;
  let open_phase = 0.65;
  if (phase < open_phase) {
    let p = phase / open_phase;
    glottal = (3.0 * p * p - 2.0 * p * p * p);
  }

  let pi2 = 6.28318530718;
  let t_sec = f32(idx) / sr;
  let formant_mod1 = sin(pi2 * params.f1 * t_sec);
  let formant_mod2 = sin(pi2 * params.f2 * t_sec) * 0.5;
  let formant_mod3 = sin(pi2 * params.f3 * t_sec) * 0.25;

  let filtered = glottal * (0.3 + 0.4 * formant_mod1 + 0.2 * formant_mod2 + 0.1 * formant_mod3);

  let total_f = f32(params.total_samples);
  let progress = f32(idx) / total_f;
  var env = 1.0;
  if (progress < 0.05) {
    env = progress / 0.05;
  } else if (progress > 0.85) {
    env = (1.0 - progress) / 0.15;
  }

  output_pcm[idx] = clamp(filtered * env * 0.6, -1.0, 1.0);
}
`;
