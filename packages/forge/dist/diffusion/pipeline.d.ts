/**
 * 파일 생성일: 2026-09-03
 * 수정일: 2026-09-03 (가짜 decay 수식 완전 적출, UNET_FORWARD_NOT_IMPLEMENTED 즉시 분출)
 * AMEVA-Forge Release 3.0: SCRUM-330 In-Browser WebGPU Diffusion Pipeline Orchestrator
 *
 * WHAT: 디퓨전 파이프라인의 전체 컴포넌트(스케줄러, UNet, VAE, 텍스트 인코더)의 생명주기를 관리하는 오케스트레이터입니다.
 * WHY: 가짜 감쇠 수식(decay = 1/(1+step*0.5))으로 UNet 순전파를 속이는 기만 행위를 원천 박멸하고,
 *      UNet 그래프가 연결되지 않은 상태에서 호출될 경우 즉각 예외를 분출(Fail-Fast)하기 위해 존재합니다.
 * HOW: 스케줄러(EulerDiscreteScheduler)와 VAE(VAEDecoder)는 연동 준비되었으나,
 *      UNet 순전파 그래프가 탑재되기 전까지는 generate() 시 UNET_FORWARD_NOT_IMPLEMENTED 에러를 즉각 던집니다.
 */
import { GGUFHeader } from '../loader/ggufStreamer';
import { EulerDiscreteScheduler } from './scheduler';
import { DecodedImage, VAEDecoderWeights } from './vaeDecoder';
export declare enum DiffusionPipelineErrorCode {
    UNET_FORWARD_NOT_IMPLEMENTED = "UNET_FORWARD_NOT_IMPLEMENTED",
    CLIP_ENCODER_NOT_IMPLEMENTED = "CLIP_ENCODER_NOT_IMPLEMENTED",
    VAE_WEIGHTS_REQUIRED = "VAE_WEIGHTS_REQUIRED",
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
}
export declare class DiffusionPipelineError extends Error {
    readonly code: DiffusionPipelineErrorCode;
    constructor(code: DiffusionPipelineErrorCode, message: string, options?: {
        cause?: unknown;
    });
}
export interface GenerationOptions {
    prompt: string;
    negativePrompt?: string;
    numSteps?: number;
    width?: number;
    height?: number;
    seed?: number;
    guidanceScale?: number;
    vaeWeights?: VAEDecoderWeights;
    onProgress?: (progress: GenerationProgress) => void;
}
export interface GenerationProgress {
    step: number;
    totalSteps: number;
    percentage: number;
    elapsedMs: number;
}
export declare class WebGPUDiffusionPipeline {
    modelHeader?: GGUFHeader;
    scheduler: EulerDiscreteScheduler;
    isModelLoaded: boolean;
    constructor();
    /**
     * GGUF 모델 헤더를 로드하고 가중치 오프셋 테이블을 구축합니다.
     */
    loadModel(headerBuffer: ArrayBuffer): Promise<GGUFHeader>;
    /**
     * 텍스트 프롬프트로부터 이미지를 생성합니다.
     * UNet 디노이징 신경망 그래프가 아직 완전히 연결되지 않았으므로, 가짜 decay 수식 대신
     * 즉시 UNET_FORWARD_NOT_IMPLEMENTED 에러를 던져 침묵 기만을 원천 차단합니다.
     */
    generate(options: GenerationOptions): Promise<DecodedImage>;
}
