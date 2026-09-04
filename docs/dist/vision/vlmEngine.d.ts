/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: Multimodal VLM Projector Engine
 *
 * WHAT: 비전 패치 임베딩 [N, 768]을 언어 모델(LLM) 텍스트 임베딩 공간 [N, textDim]으로 매핑하는 멀티모달 프로젝터입니다.
 * WHY: 이미지를 본 후 LLM이 그 내용을 텍스트로 추론하여 자연어로 답변할 수 있도록 시각-언어 공간을 정렬합니다.
 */
export interface VLMProjectorWeights {
    mlp1Weight: Float32Array;
    mlp1Bias?: Float32Array;
    mlp2Weight: Float32Array;
    mlp2Bias?: Float32Array;
}
export declare class VLMProjector {
    /**
     * 2-Layer GeLU MLP 프로젝터 순전파
     */
    static project(visualTokens: Float32Array, numTokens: number, weights: VLMProjectorWeights, hiddenDim?: number, llmDim?: number): Float32Array;
}
