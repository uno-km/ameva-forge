/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-222 High-Throughput LLM Generation Sampling Engine
 *
 * WHAT: Top-K, Top-P (Nucleus), Temperature, Greedy 전략을 지원하는 온디바이스 토큰 샘플러입니다.
 * WHY: LLM 추론 파이프라인의 마지막 단계에서 다음 생성 토큰(Next Token)을 높은 확률적 다양성과 제약 조건 하에서 결정론적/비결정론적으로 선택하기 위해 존재합니다.
 * HOW: 로짓(Logits) 배열을 받아 Temperature 스케일링 -> Top-K 필터링 -> Softmax 확률 변환 -> Top-P 누적 확률 컷오프 -> 카테고리 분포 샘플링을 수행합니다.
 */

export interface SamplingOptions {
  temperature?: number; // 기본값 1.0 (0이면 Greedy Argmax)
  top_k?: number;       // 상위 K개 후보 제한 (0 또는 undefined 시 비활성화)
  top_p?: number;       // 상위 P 누적 확률 컷오프 (예: 0.9, 1.0 시 비활성화)
  seed?: number;        // 결정론적 난수 시드 (Xorshift32 PRNG)
}

/**
 * Xorshift32 재현 가능한 고속 의사 난수 생성기
 */
function createPRNG(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

export class LLMSampler {
  /**
   * 로짓(Logits) 벡터로부터 다음 토큰 ID를 샘플링합니다.
   * 무할당(Zero-Allocation) 및 고속 인덱스 스왑 기반으로 동작하여 V8 Major GC를 원천 방지합니다.
   */
  public static sample(logits: Float32Array, options: SamplingOptions = {}): number {
    const vocabSize = logits.length;
    if (vocabSize === 0) throw new Error("Logits array cannot be empty");

    const temp = options.temperature !== undefined ? options.temperature : 1.0;
    const topK = options.top_k !== undefined ? options.top_k : 0;
    const topP = options.top_p !== undefined ? options.top_p : 1.0;
    const rng = options.seed !== undefined ? createPRNG(options.seed) : Math.random;

    // 1. Greedy Sampling (Temperature <= 1e-5)
    if (temp <= 1e-5) {
      let maxIdx = 0;
      let maxVal = logits[0];
      for (let i = 1; i < vocabSize; i++) {
        if (logits[i] > maxVal) {
          maxVal = logits[i];
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    // 2. 인덱스 배열 초기화 (Int32Array 단일 할당)
    const effectiveK = (topK > 0 && topK < vocabSize) ? topK : vocabSize;
    const indices = new Int32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) indices[i] = i;

    // 3. Top-K Selection (Top-K가 작을 경우 Partial Selection Sort로 O(K*V) 처리)
    if (effectiveK < vocabSize && effectiveK <= 128) {
      for (let i = 0; i < effectiveK; i++) {
        let maxPos = i;
        let maxVal = logits[indices[i]];
        for (let j = i + 1; j < vocabSize; j++) {
          if (logits[indices[j]] > maxVal) {
            maxVal = logits[indices[j]];
            maxPos = j;
          }
        }
        // Swap
        const tmp = indices[i];
        indices[i] = indices[maxPos];
        indices[maxPos] = tmp;
      }
    } else {
      // 전체 정렬
      indices.sort((a, b) => logits[b] - logits[a]);
    }

    // 4. Softmax Probability Computation over Candidates
    const maxLogit = logits[indices[0]] / temp;
    let sumExp = 0.0;
    const candidateProbs = new Float32Array(effectiveK);

    for (let i = 0; i < effectiveK; i++) {
      const p = Math.exp((logits[indices[i]] / temp) - maxLogit);
      candidateProbs[i] = p;
      sumExp += p;
    }

    const invSum = 1.0 / Math.max(sumExp, 1e-12);
    for (let i = 0; i < effectiveK; i++) {
      candidateProbs[i] *= invSum;
    }

    // 5. Top-P (Nucleus) Cutoff
    let cutoffIdx = effectiveK;
    if (topP < 1.0) {
      let cumsum = 0.0;
      for (let i = 0; i < effectiveK; i++) {
        cumsum += candidateProbs[i];
        if (cumsum >= topP) {
          cutoffIdx = i + 1;
          break;
        }
      }
    }

    // Filtered Probabilities Re-normalization
    let filteredSum = 0.0;
    for (let i = 0; i < cutoffIdx; i++) filteredSum += candidateProbs[i];
    const invFilteredSum = 1.0 / Math.max(filteredSum, 1e-12);

    // 6. Categorical Random Sampling using PRNG
    const rand = rng();
    let acc = 0.0;
    for (let i = 0; i < cutoffIdx; i++) {
      acc += candidateProbs[i] * invFilteredSum;
      if (rand <= acc) {
        return indices[i];
      }
    }

    return indices[cutoffIdx - 1];
  }
}

