/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-342 High-Precision LLM Logits Sampler
 *
 * WHAT: LLM 로짓 분포(Logits)로부터 Temperature, Top-K, Top-P(Nucleus), Repetition Penalty를
 *      수치 안정성을 유지하며 적용하여 다음 토큰 ID를 결정하는 단정밀도 샘플러입니다.
 * WHY: 침묵 결정론적 고정 출력이나 비정상 확률 폭발(NaN/Inf)을 원천 차단하고,
 *      생성 텍스트의 다양성과 문맥 일관성을 정밀하게 제어하기 위함입니다.
 * HOW: Repetition Penalty -> Temperature Scaling -> Top-K Truncation -> Softmax with Shift -> Top-P Cumulative Mask -> CDF Sampling.
 */

export interface SamplingOptions {
  temperature?: number;         // 0.0 (greedy) ~ 2.0 (high entropy)
  topK?: number;                // Keep only top K highest probability logits (e.g. 40, 50)
  topP?: number;                // Nucleus sampling cumulative probability threshold (0.0 ~ 1.0, e.g. 0.9)
  repetitionPenalty?: number;   // 1.0 = no penalty, > 1.0 = penalize repeated tokens
  seed?: number;                // Deterministic pseudo-random seed
}

export class Sampler {
  /**
   * 로짓 배열로부터 SamplingOptions를 적용하여 단일 다음 토큰 ID를 샘플링합니다.
   */
  public static sampleToken(
    logits: Float32Array,
    contextTokens: number[],
    options: SamplingOptions = {}
  ): number {
    const {
      temperature = 0.7,
      topK = 40,
      topP = 0.9,
      repetitionPenalty = 1.1,
    } = options;

    const vocabSize = logits.length;
    const workingLogits = new Float32Array(logits);

    // 1. Repetition Penalty 적용
    if (repetitionPenalty !== 1.0 && contextTokens.length > 0) {
      const seen = new Set(contextTokens);
      for (const tok of seen) {
        if (tok >= 0 && tok < vocabSize) {
          if (workingLogits[tok] > 0) {
            workingLogits[tok] /= repetitionPenalty;
          } else {
            workingLogits[tok] *= repetitionPenalty;
          }
        }
      }
    }

    // 2. Greedy Sampling (Temperature == 0.0)
    if (temperature <= 1e-5) {
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        if (workingLogits[i] > maxVal) {
          maxVal = workingLogits[i];
          maxIdx = i;
        }
      }
      return maxIdx;
    }

    // 3. Temperature Scaling
    for (let i = 0; i < vocabSize; i++) {
      workingLogits[i] /= temperature;
    }

    // 4. Top-K Filtering
    let effectiveK = Math.min(topK > 0 ? topK : vocabSize, vocabSize);
    // Index-Value 쌍 생성
    const indexed: { index: number; value: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
      indexed.push({ index: i, value: workingLogits[i] });
    }
    // Partial sort for Top-K
    indexed.sort((a, b) => b.value - a.value);
    const topCandidates = indexed.slice(0, effectiveK);

    // 5. Numerically Stable Softmax (Shifted Exp)
    const maxLogit = topCandidates[0].value;
    let sumExp = 0.0;
    const probs = new Float32Array(topCandidates.length);

    for (let i = 0; i < topCandidates.length; i++) {
      const p = Math.exp(topCandidates[i].value - maxLogit);
      probs[i] = p;
      sumExp += p;
    }

    const invSum = 1.0 / (sumExp + 1e-9);
    for (let i = 0; i < topCandidates.length; i++) {
      probs[i] *= invSum;
    }

    // 6. Top-P (Nucleus) Filtering
    let cumProb = 0.0;
    let cutoffIdx = topCandidates.length;
    for (let i = 0; i < topCandidates.length; i++) {
      cumProb += probs[i];
      if (cumProb >= topP && i >= 1) {
        cutoffIdx = i + 1;
        break;
      }
    }

    // 7. Final Categorical Sampling over Cutoff Candidates
    let reSum = 0.0;
    for (let i = 0; i < cutoffIdx; i++) {
      reSum += probs[i];
    }
    const invReSum = 1.0 / (reSum + 1e-9);

    const r = Math.random();
    let acc = 0.0;
    for (let i = 0; i < cutoffIdx; i++) {
      acc += probs[i] * invReSum;
      if (r <= acc) {
        return topCandidates[i].index;
      }
    }

    return topCandidates[0].index;
  }
}
