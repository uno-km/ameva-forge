/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const COL2IM_WGSL = `
/**
 * @struct Params
 * @brief 합성곱(Convolution) 연산의 역전파 과정에서 필요한 col2im (Column to Image) 연산용 파라미터 구조체입니다. (What)
 * im2col을 통해 펼쳐진 행렬 형태의 그레이디언트를 다시 원래 텐서(이미지) 형태로 복원하기 위한 정보를 담고 있습니다. (Why)
 */
struct Params {
  // 배치 크기 (Batch size)
  N: u32,
  // 채널의 개수 (Channels)
  C: u32,
  // 원본 입력 텐서의 높이 (Height)
  H: u32,
  // 원본 입력 텐서의 너비 (Width)
  W: u32,
  // 합성곱 커널의 높이 크기
  K_h: u32,
  // 합성곱 커널의 너비 크기
  K_w: u32,
  // 필터 이동 보폭 (Stride)
  stride: u32,
  // 텐서 테두리에 덧붙인 패딩 크기
  padding: u32,
  // 합성곱 연산 결과 출력 텐서의 높이
  H_out: u32,
  // 합성곱 연산 결과 출력 텐서의 너비
  W_out: u32,
  // 2D 디스패치 선형 인덱스 복원을 위한 X축 워크그룹 수입니다.
  workgroups_x: u32,
  pad1: u32,
};

// params: col2im 역산 및 복원 계산을 위한 각종 텐서 차원들을 포함한 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// grad_x_col: im2col 형태로 전개되어 있던 그레이디언트 1차원 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> grad_x_col: array<f32>;
// grad_x: 다시 원래 이미지 크기(N, C, H, W)로 합산 복원될 입력 텐서에 대한 그레이디언트 배열입니다.
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;

/**
 * @function main
 * @brief 컴퓨트 셰이더의 메인 함수로, 원본 이미지의 픽셀 인덱스별로 연관되었던 모든 커널 윈도우들의 기울기(gradient)를 합산(accumulate)합니다. (What)
 * CNN 합성곱 층에서 입력값에 대한 역전파(Backpropagation)를 수행하여 가중치 갱신에 필요한 값을 도출하기 위해 (Why) 작성되었습니다.
 * 
 * @param global_id 워크그룹 내 스레드의 3차원 전역 인덱스입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 2D 디스패치 그리드로부터 복원한 현재 스레드가 처리할 원본 텐서 상의 1차원 인덱스입니다. (How)
  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;
  // 전체 요소 개수 = 배치 * 채널 * 높이 * 너비 를 계산합니다.
  let num_elements = params.N * params.C * params.H * params.W;
  
  // 인덱스가 전체 크기를 벗어나면 즉시 함수를 빠져나가(return) 오류를 막습니다. (Why)
  if (idx >= num_elements) { return; }

  // 1차원 인덱스 idx를 4차원 좌표인 (n, c, h, w)로 복원하기 위한 임시 변수입니다. (How)
  var temp = idx;
  // 너비 차원 (Width) 복원
  let w = temp % params.W;
  temp = temp / params.W;
  // 높이 차원 (Height) 복원
  let h = temp % params.H;
  temp = temp / params.H;
  // 채널 차원 (Channel) 복원
  let c = temp % params.C;
  // 배치 차원 (Batch) 복원
  let n = temp / params.C;

  // 원본 텐서의 특정 픽셀 (n, c, h, w)에 모여들 그레이디언트 값을 누적하기 위한 실수형 변수입니다. (What)
  var val = 0.0;
  
  // 커널의 높이(K_h)만큼 반복하며 이 픽셀에 영향을 주었던 합성곱 윈도우들을 역추적합니다. (How)
  for (var k_h = 0u; k_h < params.K_h; k_h = k_h + 1u) {
    // 패딩이 적용된 높이 좌표를 계산합니다. (What)
    let h_plus_pad = h + params.padding;
    
    // 현재 커널 인덱스 k_h보다 크거나 같은지 검사하여 필터 범위를 벗어나지 않았는지 판단합니다. (Why)
    if (h_plus_pad >= k_h) {
      // 커널 내부에서의 오프셋을 제거하여 원본 인덱스를 역계산합니다. (How)
      let h_rem = h_plus_pad - k_h;
      // 스트라이드(stride) 조건에 맞게 정확하게 나누어 떨어지는 윈도우 위치인지 검사합니다. (What)
      if (h_rem % params.stride == 0u) {
        // 출력 텐서 상의 y좌표(h_out)를 복원 계산합니다.
        let h_out = h_rem / params.stride;
        // 계산된 출력 좌표가 실제 출력 텐서의 높이 범위 내에 있는지 검사합니다.
        if (h_out < params.H_out) {
          
          // 커널의 너비(K_w)만큼 반복하며 수평 방향 윈도우들을 탐색합니다. (How)
          for (var k_w = 0u; k_w < params.K_w; k_w = k_w + 1u) {
            // 패딩이 적용된 너비 좌표를 계산합니다.
            let w_plus_pad = w + params.padding;
            
            // 현재 커널 인덱스 k_w보다 크거나 같은지 확인하여 유효 범위인지 검사합니다.
            if (w_plus_pad >= k_w) {
              // 커널 너비 내의 오프셋을 제거합니다.
              let w_rem = w_plus_pad - k_w;
              // 수평 스트라이드 조건에 정확히 부합하는지 확인합니다. (What)
              if (w_rem % params.stride == 0u) {
                // 출력 텐서 상의 x좌표(w_out)를 계산합니다. (How)
                let w_out = w_rem / params.stride;
                // 계산된 출력 좌표가 실제 출력 텐서 너비 범위에 들어오는지 검증합니다.
                if (w_out < params.W_out) {
                  // 배치 번호는 원본과 동일하게 가져옵니다.
                  let n_out = n;
                  // 출력 평면 2D 상의 1차원 선형 인덱스(hw_out)를 계산합니다. (How)
                  let hw_out = h_out * params.W_out + w_out;
                  // 커널 안에서의 채널 및 2D 윈도우 인덱스(c_kw_kh)를 1차원으로 계산합니다.
                  let c_kw_kh = (c * params.K_h + k_h) * params.K_w + k_w;
                  
                  // 위에서 계산한 값들을 바탕으로, 2차원으로 전개되었던 grad_x_col 배열의 실제 접근 인덱스를 합성합니다. (What)
                  let col_idx = (n_out * (params.H_out * params.W_out) + hw_out) * (params.C * params.K_h * params.K_w) + c_kw_kh;
                  // 전개된 배열에서 가져온 기울기(gradient) 값을 현재 픽셀의 누적기(val)에 더합니다. (How)
                  val = val + grad_x_col[col_idx];
                }
              }
            }
          }
          
        }
      }
    }
  }

  // 역추적된 윈도우들로부터 누적 계산이 모두 끝난 총 그레이디언트 값을 출력 배열에 저장합니다. (What)
  grad_x[idx] = val;
}
`;
