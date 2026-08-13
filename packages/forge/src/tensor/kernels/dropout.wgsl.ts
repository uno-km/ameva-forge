/**
 * 생성일 (Created): 2026-08-12 12:59:35 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 */
export const DROPOUT_WGSL = `
/**
 * @struct Params
 * @brief 드롭아웃(Dropout) 연산을 수행하기 위해 필요한 메타데이터를 저장하는 구조체입니다. (What)
 * 과적합(Overfitting) 방지를 위해 무작위로 뉴런(값)을 0으로 끄는 확률(p)과 난수 시드(seed) 정보를 GPU에 전달하기 위해 사용됩니다. (Why)
 */
struct Params {
  // 드롭아웃을 적용할 전체 데이터 원소의 개수입니다.
  num_elements: u32,
  // 난수 생성의 기반이 되는 시드(seed) 값입니다. 매번 다른 패턴의 드롭아웃을 적용하기 위해 외부에서 주입됩니다.
  seed: f32,
  // 드롭아웃 확률(p)입니다. (0.0 ~ 1.0) 이 확률보다 낮게 난수가 나오면 해당 값을 0으로 끕니다.
  p: f32,
  // 데이터 정렬 규칙(16바이트)을 충족시키기 위해 존재하는 의미 없는 패딩 값입니다.
  padding: f32,
}

// params: 드롭아웃 파라미터를 담고 있는 Uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// x: 입력 데이터를 보관하고 있는 텐서(1차원 배열)입니다. (읽기 전용)
@group(0) @binding(1) var<storage, read> x: array<f32>;
// out: 드롭아웃 적용 이후 결과가 저장될 출력 데이터 텐서입니다.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

/**
 * @function pcg_hash
 * @brief PCG (Permuted Congruential Generator) 기반의 해시 함수를 통해 32비트 정수형 난수를 생성합니다. (What)
 * 셰이더 내부에는 내장된 난수 생성기가 없으므로, 인덱스와 시드를 바탕으로 빠르고 균일하게 의사 난수(Pseudo Random)를 만들기 위해 고안되었습니다. (Why)
 * @param input 난수의 입력이 되는 시드 역할의 부호 없는 정수입니다. (How)
 * @return 해시 변환된 새로운 32비트 난수(u32)를 반환합니다.
 */
fn pcg_hash(input: u32) -> u32 {
    // 입력된 정수에 큰 소수를 곱하고 상수를 더해 초기 상태(state)를 섞습니다. (How)
    var state = input * 747796405u + 2891336453u;
    // 비트 시프트 연산과 XOR을 통해 비트 패턴을 비선형적으로 한 번 더 혼합합니다. (How)
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    // 최종적으로 비트 이동 후 XOR하여 고품질의 난수를 반환합니다.
    return (word >> 22u) ^ word;
}

/**
 * @function rand_f32
 * @brief 32비트 정수 형태의 난수를 0.0 이상 1.0 미만의 부동소수점(float) 형태로 정규화합니다. (What)
 * 드롭아웃 확률(p)과 직접 크기를 비교하기 위해 0~1 사이의 값이 필요하기 때문입니다. (Why)
 * @param hash pcg_hash로부터 전달받은 32비트 무작위 정수입니다.
 * @return 0.0과 1.0 사이로 매핑된 실수 난수입니다. (How)
 */
fn rand_f32(hash: u32) -> f32 {
    // 32비트 정수의 최대값(4294967295)으로 나누어 0~1 범위의 실수로 변환합니다. (How)
    return f32(hash) / 4294967295.0;
}

/**
 * @function main
 * @brief 스레드별로 난수를 발생시켜 지정된 확률 p 미만이면 0을, 그 이상이면 스케일링된 원본 값을 출력 텐서에 기록합니다. (What)
 * 신경망 학습 시 특정 노드에 과도하게 의존하는 현상을 막기 위해(Why) 병렬 스레드를 이용하여 고속으로 무작위 노드 비활성화를 수행합니다.
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2차원(Y) 워크그룹 구조를 1차원으로 풀어 현재 스레드의 전역 선형 인덱스를 계산합니다. (How)
    // 참고로 65535u는 X방향 워크그룹의 최대 한계를 가정하여 하드코딩된 변환 값입니다.
    let index = global_id.x + global_id.y * 65535u * 64u;
    
    // 계산된 인덱스가 전체 텐서의 원소 수보다 크거나 같으면 실행을 즉시 중단합니다. (What)
    // 올바르지 않은 메모리 범위를 건드리지 않도록 차단하는 역할입니다. (Why)
    if (index >= params.num_elements) {
        return;
    }
    
    // 현재 인덱스와 외부에서 입력받은 시드를 조합(스케일업)하여 해시 생성기의 입력(input)을 구성하고 난수 정수를 만듭니다. (How)
    let hash = pcg_hash(index + u32(params.seed * 10000.0));
    // 정수 형태의 해시를 0.0 ~ 1.0 사이의 실수 난수로 변환합니다. (How)
    let rand = rand_f32(hash);
    
    // 생성된 난수가 설정된 드롭아웃 확률 p보다 작은지 검사합니다. (What)
    if (rand < params.p) {
        // 확률 분포에 걸렸을 경우(노드 비활성화), 해당 인덱스의 출력값을 0.0으로 만듭니다. (How)
        out[index] = 0.0;
    } else {
        // 확률 분포에 걸리지 않은 경우, 원본 데이터를 그대로 유지하되 기대값을 보존하기 위해 1/(1-p) 만큼 스케일링(Scaling)하여 저장합니다. (How)
        // 이는 Inverted Dropout 기법으로, 테스트 단계에서 별도의 스케일링 작업 없이 바로 모델을 쓸 수 있게 만들기 위함입니다. (Why)
        out[index] = x[index] * (1.0 / (1.0 - params.p));
    }
}
`;
