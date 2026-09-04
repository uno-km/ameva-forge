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
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
    seed?: number;
}
export declare class Sampler {
    /**
     * 로짓 배열로부터 SamplingOptions를 적용하여 단일 다음 토큰 ID를 샘플링합니다.
     */
    static sampleToken(logits: Float32Array, contextTokens: number[], options?: SamplingOptions): number;
}
