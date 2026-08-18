/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-234 INT4 / INT8 Quantized Weight Unpacking WGSL Kernel
 *
 * WHAT: GGUF(Q4_K_M/Q8_0) 및 AWQ/GPTQ 양자화된 신경망 가중치를 GPU 상에서 실시간 FP32로 언패킹하는 WGSL 셰이더입니다.
 * WHY: 7B LLM 가중치를 4GB 미만으로 브라우저에 로드하고, 메모리 대역폭을 75% 절감하여 초고속 추론을 달성하기 위해 존재합니다.
 * HOW: u32 정수에 패킹된 8개의 4-bit 값(또는 4개의 8-bit 값)을 비트 시프트 및 마스킹으로 추출하고,
 *      scale과 zero_point를 적용하여 역양자화(Dequantization)합니다.
 */

export const UNPACK_QUANT_WGSL = `
struct Params {
  num_elements: u32,    // 총 복원될 원소 개수 (FP32 개수)
  bits: u32,            // 4 또는 8
  group_size: u32,      // 양자화 그룹 크기 (예: 32, 128)
  workgroupsX: u32,     // 2D 디스패치 X 크기
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> packed_data: array<u32>; // 패킹된 정수 배열
@group(0) @binding(2) var<storage, read> scales: array<f32>;      // 그룹별 스케일
@group(0) @binding(3) var<storage, read> zeros: array<f32>;       // 그룹별 제로포인트
@group(0) @binding(4) var<storage, read_write> out_fp32: array<f32>; // 복원된 FP32 배열

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let idx = (workgroup_id.x + workgroup_id.y * params.workgroupsX) * 64u + local_id.x;

  if (idx >= params.num_elements) {
    return;
  }

  let group_idx = idx / params.group_size;
  let scale = scales[group_idx];
  let zero = zeros[group_idx];

  var raw_int: f32 = 0.0;

  if (params.bits == 4u) {
    // 4-bit 언패킹: 1개 u32에 8개 니블(nibble) 저장
    let word_idx = idx / 8u;
    let nibble_idx = idx % 8u;
    let shift = nibble_idx * 4u;
    let packed_val = packed_data[word_idx];
    let val_4bit = (packed_val >> shift) & 0x0Fu;
    raw_int = f32(val_4bit);
  } else if (params.bits == 8u) {
    // 8-bit 언패킹: 1개 u32에 4개 바이트 저장
    let word_idx = idx / 4u;
    let byte_idx = idx % 4u;
    let shift = byte_idx * 8u;
    let packed_val = packed_data[word_idx];
    let val_8bit = (packed_val >> shift) & 0xFFu;
    raw_int = f32(val_8bit);
  }

  // Dequantize: (int_val - zero) * scale
  out_fp32[idx] = (raw_int - zero) * scale;
}
`;
