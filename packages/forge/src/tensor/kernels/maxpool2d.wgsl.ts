/**
 * 파일 생성일: 2026-08-12 12:59:35 +0900 (commit 67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
 * 수정 이력:
 * - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const MAXPOOL2D_WGSL = `
/**
 * 이 구조체(Params)는 2D 맥스 풀링(Max Pooling 2D) 연산에 필요한 하이퍼파라미터 및 차원 정보를 담고 있습니다.
 * 입력 이미지의 배치, 채널, 크기 정보와 커널 크기, 스트라이드, 패딩 값을 전달하기 위해 존재합니다.
 */
struct Params {
    batch: u32, // 배치 크기입니다.
    channels: u32, // 입력 텐서의 채널 수입니다.
    in_h: u32, // 원본 입력 이미지의 높이입니다.
    in_w: u32, // 원본 입력 이미지의 너비입니다.
    out_h: u32, // 계산되어 출력될 이미지의 높이입니다.
    out_w: u32, // 계산되어 출력될 이미지의 너비입니다.
    kH: u32, // 풀링 커널(필터)의 높이입니다.
    kW: u32, // 풀링 커널(필터)의 너비입니다.
    sH: u32, // 높이 방향의 스트라이드(보폭)입니다.
    sW: u32, // 너비 방향의 스트라이드(보폭)입니다.
    pH: u32, // 높이 방향에 추가된 제로 패딩 크기입니다.
    pW: u32, // 너비 방향에 추가된 제로 패딩 크기입니다.
}

@group(0) @binding(0) var<uniform> params: Params; // GPU에 메타데이터를 공급하는 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // NCHW 형태의 입력 데이터 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 풀링 결과가 저장될 NCHW 형태의 출력 텐서입니다.

/**
 * main 함수는 합성곱 신경망(CNN)의 핵심 구성 요소인 2D 맥스 풀링 연산을 수행합니다.
 * 이미지의 국소 영역(커널 크기)에서 최댓값만을 추출하여 공간적 차원(Spatial Dimension)을 축소하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // 출력 텐서에서 현재 스레드가 담당할 1D 위치(인덱스)입니다.
    // 연산이 필요한 전체 출력 데이터의 개수를 계산합니다.
    let total = params.batch * params.channels * params.out_h * params.out_w;
    
    // 할당된 스레드 인덱스가 유효 범위를 벗어나면 연산을 중단합니다.
    if (idx >= total) {
        return;
    }
    
    // 1D 인덱스에서 NCHW 포맷에 따라 출력 좌표 (ow, oh, c, b)를 역산합니다.
    let ow = idx % params.out_w; // 출력 맵의 너비(x) 좌표입니다.
    let oh = (idx / params.out_w) % params.out_h; // 출력 맵의 높이(y) 좌표입니다.
    let c = (idx / (params.out_w * params.out_h)) % params.channels; // 채널 인덱스입니다.
    let b = idx / (params.out_w * params.out_h * params.channels); // 배치 인덱스입니다.
    
    // 스트라이드와 패딩을 적용하여 입력 이미지 기준 시작 좌표를 계산합니다.
    let h_start = i32(oh * params.sH) - i32(params.pH);
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    // 최댓값 비교를 위한 초기값을 부동소수점 표현 가능한 가장 작은 값으로 설정합니다.
    var max_val = -3.402823466e+38; // -FLT_MAX
    
    // 커널의 높이(kH)와 너비(kW) 영역을 순회하며 최댓값을 찾기 위한 이중 루프입니다.
    for (var kh = 0u; kh < params.kH; kh++) {
        for (var kw = 0u; kw < params.kW; kw++) {
            // 커널 내 오프셋을 더하여 실제 입력 데이터 상의 좌표를 구합니다.
            let h = h_start + i32(kh);
            let w = w_start + i32(kw);
            
            // 계산된 좌표가 이미지 경계 안쪽에 있는지(유효한 데이터인지) 검사합니다.
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                // NCHW 포맷에 따른 입력 텐서의 1D 메모리 인덱스를 계산합니다.
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                let val = input[in_idx]; // 입력값을 읽어옵니다.
                
                // 기존의 max_val과 비교하여 더 큰 값이면 갱신합니다.
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // 커널 영역 전체에서 발견한 최댓값을 출력 텐서의 현재 인덱스에 저장합니다.
    output[idx] = max_val;
}
`;
