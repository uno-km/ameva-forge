/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-222 High-Throughput LLM Generation Sampling Engine
 *
 * WHAT: Top-K, Top-P (Nucleus), Temperature, Greedy 전략을 지원하는 온디바이스 토큰 샘플러입니다.
 * WHY: LLM 추론 파이프라인의 마지막 단계에서 다음 생성 토큰(Next Token)을 높은 확률적 다양성과 제약 조건 하에서 결정론적/비결정론적으로 선택하기 위해 존재합니다.
 * HOW: 로짓(Logits) 배열을 받아 Temperature 스케일링 -> Top-K 필터링 -> Softmax 확률 변환 -> Top-P 누적 확률 컷오프 -> 카테고리 분포 샘플링을 수행합니다.
 */
export interface SamplingOptions {
    temperature?: number;
    top_k?: number;
    top_p?: number;
    seed?: number;
}
export declare class LLMSampler {
    /**
     * 로짓(Logits) 벡터로부터 다음 토큰 ID를 샘플링합니다.
     * 무할당(Zero-Allocation) 및 고속 인덱스 스왑 기반으로 동작하여 V8 Major GC를 원천 방지합니다.
     */
    static sample(logits: Float32Array, options?: SamplingOptions): number;
}
