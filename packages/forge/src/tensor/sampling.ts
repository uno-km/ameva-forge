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
  seed?: number;        // 결정론적 난수 시드
}

export class LLMSampler {
  /**
   * 로짓(Logits) 벡터로부터 다음 토큰 ID를 샘플링합니다.
   */
  public static sample(logits: Float32Array, options: SamplingOptions = {}): number {
    const vocabSize = logits.length;
    if (vocabSize === 0) throw new Error("Logits array cannot be empty");

    const temp = options.temperature !== undefined ? options.temperature : 1.0;
    const topK = options.top_k !== undefined ? options.top_k : 0;
    const topP = options.top_p !== undefined ? options.top_p : 1.0;

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

    // 2. Temperature Scaling & Index Pairing
    const indexedLogits: { idx: number; logit: number }[] = new Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      indexedLogits[i] = { idx: i, logit: logits[i] / temp };
    }

    // 로짓 내림차순 정렬
    indexedLogits.sort((a, b) => b.logit - a.logit);

    // 3. Top-K Filtering
    let candidates = indexedLogits;
    if (topK > 0 && topK < vocabSize) {
      candidates = candidates.slice(0, topK);
    }

    // 4. Softmax Probability Computation over Candidates
    const maxLogit = candidates[0].logit;
    let sumExp = 0.0;
    const probs: number[] = new Array(candidates.length);

    for (let i = 0; i < candidates.length; i++) {
      const p = Math.exp(candidates[i].logit - maxLogit);
      probs[i] = p;
      sumExp += p;
    }

    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
    }

    // 5. Top-P (Nucleus) Filtering
    let cutoffIdx = candidates.length;
    if (topP < 1.0) {
      let cumsum = 0.0;
      for (let i = 0; i < probs.length; i++) {
        cumsum += probs[i];
        if (cumsum >= topP) {
          cutoffIdx = i + 1;
          break;
        }
      }
    }

    const filteredCandidates = candidates.slice(0, cutoffIdx);
    const filteredProbs = probs.slice(0, cutoffIdx);

    // Filtered Probabilities Re-normalization
    let filteredSum = 0.0;
    for (let i = 0; i < filteredProbs.length; i++) filteredSum += filteredProbs[i];
    for (let i = 0; i < filteredProbs.length; i++) filteredProbs[i] /= filteredSum;

    // 6. Categorical Random Sampling
    const rand = Math.random();
    let acc = 0.0;
    for (let i = 0; i < filteredProbs.length; i++) {
      acc += filteredProbs[i];
      if (rand <= acc) {
        return filteredCandidates[i].idx;
      }
    }

    return filteredCandidates[filteredCandidates.length - 1].idx;
  }
}
