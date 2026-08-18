/**
 * 파일 생성일: 2026-08-12 12:59:35 +0900 (commit 67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
 * 수정 이력:
 * - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const IM2COL_WGSL = `
/**
 * 이 구조체(Params)는 이미지 데이터(공간적 텐서)를 열(Column) 기반 행렬로 변환하기 위한 컨볼루션 인자들을 담고 있습니다.
 * 입력 이미지의 크기, 커널(필터)의 크기, 스트라이드, 패딩 등 im2col 연산에 필수적인 하이퍼파라미터를 제공하기 위해 존재합니다.
 */
struct Params {
  N: u32, // 배치 크기 (Batch size)입니다.
  C: u32, // 입력 채널 수 (Channels)입니다.
  H: u32, // 입력 이미지의 원본 높이 (Height)입니다.
  W: u32, // 입력 이미지의 원본 너비 (Width)입니다.
  K_h: u32, // 커널(필터)의 높이입니다.
  K_w: u32, // 커널(필터)의 너비입니다.
  stride: u32, // 합성곱 연산 시 필터가 이동하는 보폭(스트라이드)입니다.
  padding: u32, // 입력 이미지 가장자리에 추가할 제로 패딩의 크기입니다.
  H_out: u32, // 연산 후 생성될 출력 특성 맵의 높이입니다.
  W_out: u32, // 연산 후 생성될 출력 특성 맵의 너비입니다.
  workgroups_x: u32, // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수입니다.
  pad1: u32, // 16바이트 메모리 정렬을 위한 패딩입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 컨볼루션 설정값을 전달하는 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // NCHW 형태로 펼쳐진 원본 이미지 입력 배열입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 변환된 행렬 형태의 데이터가 기록될 출력 배열입니다.

/**
 * main 함수는 합성곱(Convolution) 연산을 행렬 곱(MatMul)으로 효율적으로 수행하기 위해
 * 이미지 데이터의 국소적 패치(Local patch)를 추출하여 2D 행렬 형태로 재배치(im2col)합니다.
 * 이를 통해 GPU 상에서 고속의 GEMM(General Matrix Multiply) 라이브러리 및 최적화를 활용할 수 있습니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u; // 2D 디스패치 선형 인덱스 복원
  // 변환될 출력 배열의 총 요소 개수를 계산합니다 (N * H_out * W_out * C * K_h * K_w).
  let num_elements = params.N * params.H_out * params.W_out * params.C * params.K_h * params.K_w;
  
  // 계산할 인덱스가 배열 크기를 넘어가면 실행을 종료합니다.
  if (idx >= num_elements) { return; }

  var temp = idx; // 1차원 인덱스를 다차원 인덱스로 역계산하기 위해 임시 변수에 저장합니다.
  
  // 출력 버퍼의 인덱스에서 채널 및 커널 위치에 해당하는 차원 값을 추출합니다.
  let c_kw_kh = temp % (params.C * params.K_h * params.K_w);
  temp = temp / (params.C * params.K_h * params.K_w); // 다음 차원 추출을 위해 값을 나눕니다.
  
  // 출력 특성 맵의 공간적 위치(높이, 너비) 차원 값을 추출합니다.
  let h_out_w_out = temp % (params.H_out * params.W_out);
  temp = temp / (params.H_out * params.W_out); // 다음 차원 추출을 위해 값을 나눕니다.
  
  // 최종적으로 배치(Batch) 인덱스를 추출합니다.
  let n = temp % params.N;

  // 커널 내에서의 로컬 x, y 좌표 및 채널 인덱스를 계산합니다.
  let k_w = c_kw_kh % params.K_w; // 커널 내에서의 너비 인덱스
  let k_h = (c_kw_kh / params.K_w) % params.K_h; // 커널 내에서의 높이 인덱스
  let c = c_kw_kh / (params.K_w * params.K_h); // 입력 채널 인덱스

  // 출력 특성 맵 내에서의 x, y 좌표를 계산합니다.
  let w_out = h_out_w_out % params.W_out; // 출력 맵에서의 너비 위치
  let h_out = h_out_w_out / params.W_out; // 출력 맵에서의 높이 위치

  // 커널 위치와 스트라이드, 패딩을 고려하여 원본 입력 이미지 상의 실제 y, x 좌표를 역산합니다.
  let h_in = i32(h_out * params.stride) - i32(params.padding) + i32(k_h);
  let w_in = i32(w_out * params.stride) - i32(params.padding) + i32(k_w);

  // 계산된 원본 위치가 이미지 경계 내부인지 검사합니다.
  if (h_in >= 0 && h_in < i32(params.H) && w_in >= 0 && w_in < i32(params.W)) {
    // 경계 내부라면 NCHW 포맷에 따라 입력 배열의 1D 인덱스를 계산하고 값을 가져와 저장합니다.
    let in_idx = ((n * params.C + c) * params.H + u32(h_in)) * params.W + u32(w_in);
    output[idx] = input[in_idx];
  } else {
    // 경계 밖이라면 패딩 영역이므로 0.0을 채워 넣습니다.
    output[idx] = 0.0;
  }
}
`;
