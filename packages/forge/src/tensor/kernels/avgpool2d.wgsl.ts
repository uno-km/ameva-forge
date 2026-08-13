/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const AVGPOOL2D_WGSL = `
/**
 * @struct Params
 * @brief 2D 평균 풀링(Average Pooling 2D) 연산에 필요한 하이퍼파라미터 및 텐서 차원 정보를 저장합니다. (What)
 * 셰이더 내에서 입력 텐서의 특정 영역을 순회하고 평균을 계산하기 위한 기준 값들로 사용됩니다. (Why)
 */
struct Params {
    // 배치(batch) 크기입니다. 여러 이미지를 동시에 처리하기 위한 차원입니다.
    batch: u32,
    // 채널(channel) 수입니다. 예를 들어 RGB 이미지의 경우 3이 될 수 있습니다.
    channels: u32,
    // 입력 이미지의 높이(height) 차원 크기입니다.
    in_h: u32,
    // 입력 이미지의 너비(width) 차원 크기입니다.
    in_w: u32,
    // 출력 이미지의 높이 차원 크기입니다. 연산 후의 공간적 크기를 나타냅니다.
    out_h: u32,
    // 출력 이미지의 너비 차원 크기입니다.
    out_w: u32,
    // 풀링 커널(kernel)의 높이 크기입니다.
    kH: u32,
    // 풀링 커널(kernel)의 너비 크기입니다.
    kW: u32,
    // 높이 방향의 스트라이드(stride, 이동 보폭)입니다.
    sH: u32,
    // 너비 방향의 스트라이드(stride, 이동 보폭)입니다.
    sW: u32,
    // 높이 방향의 패딩(padding) 크기입니다.
    pH: u32,
    // 너비 방향의 패딩(padding) 크기입니다.
    pW: u32,
}

// params: 연산 정보를 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// input: 풀링 연산을 수행할 원본 입력 데이터 배열(읽기 전용)입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;
// output: 풀링 연산 결과가 기록될 출력 데이터 배열입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

/**
 * @function main
 * @brief 컴퓨트 셰이더의 진입점으로, 각 스레드가 하나의 출력 픽셀에 대한 2D 평균 풀링 연산을 수행합니다. (What)
 * GPU의 수많은 스레드를 활용하여 이미지 전체 영역 및 배치 데이터를 병렬로 압축 처리하기 위해 존재합니다. (Why)
 * @param global_id 워크그룹 및 스레드의 전역적인 3차원 위치(인덱스)입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 현재 스레드의 선형 인덱스를 가져옵니다.
    let idx = global_id.x;
    
    // 계산해야 할 전체 출력 요소의 총합을 구합니다 (배치 * 채널 * 출력높이 * 출력너비). (How)
    let total = params.batch * params.channels * params.out_h * params.out_w;
    
    // 스레드 인덱스가 유효 범위를 벗어나면 즉시 함수를 종료(return)하여 잘못된 메모리 접근을 막습니다. (Why)
    if (idx >= total) {
        return;
    }
    
    // 1차원 인덱스 idx를 4차원 좌표 (b, c, oh, ow)로 변환하는 과정입니다. (How)
    // 현재 픽셀이 속한 출력 이미지의 너비 위치(ow)를 구합니다.
    let ow = idx % params.out_w;
    // 현재 픽셀이 속한 출력 이미지의 높이 위치(oh)를 구합니다.
    let oh = (idx / params.out_w) % params.out_h;
    // 현재 픽셀이 속한 채널 위치(c)를 구합니다.
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    // 현재 픽셀이 속한 배치 위치(b)를 구합니다.
    let b = idx / (params.out_w * params.out_h * params.channels);
    
    // 입력 이미지에서 현재 커널이 적용될 시작 Y좌표(높이)를 계산합니다. 패딩을 고려하여 음수가 될 수도 있습니다. (What)
    let h_start = i32(oh * params.sH) - i32(params.pH);
    // 입력 이미지에서 현재 커널이 적용될 시작 X좌표(너비)를 계산합니다. 패딩을 고려합니다.
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    // 풀링 영역 내의 픽셀 값들을 누적하기 위한 합계 변수입니다. (What)
    var sum = 0.0;
    // 풀링 영역 내에서 실제로 유효한 픽셀의 개수를 셉니다. (경계 밖은 제외하기 위함) (Why)
    var count = 0.0;
    
    // 커널의 높이만큼 반복하여 수직 방향 픽셀들을 순회합니다. (How)
    for (var kh = 0u; kh < params.kH; kh++) {
        // 커널의 너비만큼 반복하여 수평 방향 픽셀들을 순회합니다. (How)
        for (var kw = 0u; kw < params.kW; kw++) {
            // 현재 순회 중인 픽셀의 실제 입력 텐서상 Y좌표입니다.
            let h = h_start + i32(kh);
            // 현재 순회 중인 픽셀의 실제 입력 텐서상 X좌표입니다.
            let w = w_start + i32(kw);
            
            // 유효성 검사: 계산된 (h, w)가 이미지 경계를 벗어나지 않는지(0 이상, 입력 크기 미만) 확인합니다. (What)
            // 패딩 영역이나 이미지 범위를 넘어간 곳의 값은 무시하여 올바른 평균을 구하기 위함입니다. (Why)
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                // 4차원 좌표 (b, c, h, w)를 다시 1차원 인덱스(in_idx)로 변환합니다. (How)
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                
                // 해당 입력 픽셀의 값을 합산 변수에 누적시킵니다.
                sum += input[in_idx];
                // 유효한 픽셀을 한 개 처리했으므로 카운트를 증가시킵니다.
                count += 1.0;
            }
        }
    }
    
    // 유효한 픽셀 카운트가 1개 이상일 경우 정상적으로 평균을 계산합니다. (What)
    // 0으로 나누기(Division by zero) 오류를 방지하기 위함입니다. (Why)
    if (count > 0.0) {
        // 총 누적 합(sum)을 유효 픽셀 개수(count)로 나누어 평균을 구한 후, 출력 배열의 1차원 인덱스에 저장합니다. (How)
        output[idx] = sum / count;
    } else {
        // 유효한 픽셀이 전혀 없었다면(예: 모두 패딩 영역인 경우) 결과값을 0으로 처리합니다. (How)
        output[idx] = 0.0;
    }
}
`;
