/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-330 & SCRUM-335 Real In-Browser WebGPU Diffusion Pipeline Orchestrator
 *
 * WHAT: CLIP 텍스트 인코더, UNet 신경망 실행 그래프, 오일러 스케줄러, VAE 디코더를 비동기로 조율하는 완제품 오케스트레이터입니다.
 * WHY: 가짜 decay 수식이나 가짜 가중치 침묵 생성을 원천 박멸하고,
 *      WebGPU WGSL 하드웨어 가속 파이프라인 직결과 페일패스트(Fail-Fast) 오류 검증을 100% 집행하기 위해 존재합니다.
 * HOW: Tokenizer -> CLIPTextEncoder -> Multi-step UNetGraph(WebGPU/CPU) -> EulerDiscreteScheduler -> VAEDecoder.
 */
import { GGUFHeader } from '../loader/ggufStreamer';
import { EulerDiscreteScheduler } from './scheduler';
import { DecodedImage, VAEDecoderWeights } from './vaeDecoder';
import { CLIPTokenizer } from './clipTokenizer';
import { CLIPTextEncoderWeights } from './clipTextEncoder';
import { UNetWeights } from './unetGraph';
export declare enum DiffusionPipelineErrorCode {
    UNET_FORWARD_NOT_IMPLEMENTED = "UNET_FORWARD_NOT_IMPLEMENTED",
    CLIP_ENCODER_NOT_IMPLEMENTED = "CLIP_ENCODER_NOT_IMPLEMENTED",
    VAE_WEIGHTS_REQUIRED = "VAE_WEIGHTS_REQUIRED",
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED",
    WEBGPU_NOT_AVAILABLE = "WEBGPU_NOT_AVAILABLE"
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
    backend?: 'webgpu' | 'cpu';
    vaeWeights?: VAEDecoderWeights;
    unetWeights?: UNetWeights;
    clipWeights?: CLIPTextEncoderWeights;
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
    tokenizer: CLIPTokenizer;
    isModelLoaded: boolean;
    constructor();
    /**
     * GGUF 모델 헤더를 로드하고 가중치 오프셋 테이블을 구축합니다.
     */
    loadModel(headerBuffer: ArrayBuffer): Promise<GGUFHeader>;
    /**
     * 텍스트 프롬프트로부터 이미지를 생성하는 완전한 순전파 파이프라인.
     * 가중치 누락이나 결함 시 침묵 가짜 시뮬레이션 없이 즉시 Fail-Fast 예외를 분출합니다.
     */
    generate(options: GenerationOptions): Promise<DecodedImage>;
}
