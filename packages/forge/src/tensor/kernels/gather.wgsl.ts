/**
 * 파일 생성일: 2026-08-12 12:23:09 +0900 (commit fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
 * 수정 이력:
 * - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
 */
export const GATHER_WGSL = `
/**
 * 이 구조체(Params)는 gather 연산에 필요한 형태(shape), 차원(stride), 대상 차원(dim) 정보를 담고 있습니다.
 * 다차원 텐서 인덱싱을 1차원 메모리에서 올바르게 계산하기 위한 정보를 제공하기 위해 존재합니다.
 */
struct Params {
  num_elements: u32, // 출력 텐서의 총 원소 개수입니다.
  dim: u32, // 요소를 수집할(gather) 대상 차원(axis)의 인덱스입니다.
  rank: u32, // 텐서의 차원 수 (랭크)입니다.
  workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수
  x_strides: array<u32, 8>, // 원본 입력 텐서의 각 차원별 스트라이드(보폭)입니다.
  out_strides: array<u32, 8>, // 출력 텐서의 각 차원별 스트라이드(보폭)입니다.
  x_shape: array<u32, 8>, // 원본 입력 텐서의 모양(각 차원의 크기)입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 메타데이터 및 형태 정보가 담긴 유니폼 데이터입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 수집 대상이 되는 원본 데이터 배열입니다.
@group(0) @binding(2) var<storage, read> index: array<f32>; // 수집할 인덱스를 지정하는 부동소수점 배열입니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // 수집된 데이터가 쓰여질 결과 배열입니다.

/**
 * main 함수는 다차원 텐서에서 지정된 축(dim)을 기준으로 index 배열에 명시된 위치의 값들을 가져와 출력 텐서를 생성합니다.
 * PyTorch/NumPy의 gather 연산을 GPU에서 병렬 처리하기 위해 존재하며, 각 스레드는 출력 배열의 한 요소에 대응합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
  
  // 현재 처리할 인덱스가 전체 요소 수를 초과하면 실행을 중단합니다.
  if (idx >= params.num_elements) { return; }

  var temp = idx; // 다차원 좌표를 계산하기 위해 인덱스를 임시 변수에 복사합니다.
  var in_idx = 0u; // 입력 텐서에서 실제 참조해야 할 1D 메모리 인덱스를 누적할 변수입니다.

  // 출력 텐서의 각 차원(0부터 rank-1까지)에 대해 루프를 돕니다.
  // 이 루프는 출력 텐서의 1D 인덱스(idx)를 다차원 좌표로 변환하고, 이를 다시 입력 텐서의 1D 인덱스(in_idx)로 매핑합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    let out_stride = max(params.out_strides[i], 1u);
    let coord = temp / out_stride; // 현재 차원 i에서의 다차원 좌표 값입니다.
    temp = temp % out_stride; // 다음 하위 차원 좌표 계산을 위해 나머지를 구합니다.
    
    // 현재 차원이 수집 대상 차원(dim)인 경우, 계산된 좌표 대신 index 배열에서 값을 읽어옵니다.
    if (i == params.dim) {
      let raw_val = index[idx];
      if (raw_val != raw_val) {
        output[idx] = 0.0;
        return;
      }
      let dim_size = i32(params.x_shape[i]);
      var signed_idx = i32(round(raw_val));
      if (signed_idx < 0) {
        signed_idx = signed_idx + dim_size;
      }
      if (signed_idx < 0 || signed_idx >= dim_size) {
        output[idx] = 0.0;
        return;
      }
      let valid_idx = u32(signed_idx);
      in_idx = in_idx + valid_idx * params.x_strides[i];
    } else {
      // 수집 대상 차원이 아닌 경우, 출력 텐서와 동일한 좌표를 유지합니다.
      in_idx = in_idx + coord * params.x_strides[i]; // 동일한 좌표에 원본 텐서의 해당 차원 스트라이드를 곱해 누적합니다.
    }
  }

  // 최종적으로 계산된 입력 텐서 인덱스(in_idx)의 값을 읽어 출력 텐서의 현재 인덱스(idx)에 저장합니다.
  output[idx] = input[in_idx];
}
`;
