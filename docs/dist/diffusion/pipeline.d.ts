/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-330 End-to-End In-Browser WebGPU Diffusion Pipeline
 *
 * WHAT: GGUF 가중치 스트리밍 적재부터 디노이징 루프, VAE 디코딩, Canvas 렌더링까지 전체 과정을 지휘하는 통합 파이프라인입니다.
 * WHY: 연구원과 개발자가 단 3줄의 TypeScript/JS 코드로 브라우저에서 서버 없이 온디바이스 이미지 생성을 실행할 수 있도록 단일 진입점을 제공하기 위해 존재합니다.
 * HOW: GGUFStreamer -> EulerDiscreteScheduler -> ResNetBlock -> VAEDecoder 파이프라인을 비동기로 조율합니다.
 */
import { GGUFHeader } from '../loader/ggufStreamer';
import { EulerDiscreteScheduler } from './scheduler';
import { DecodedImage } from './vaeDecoder';
export interface GenerationOptions {
    prompt: string;
    negativePrompt?: string;
    numSteps?: number;
    width?: number;
    height?: number;
    seed?: number;
    guidanceScale?: number;
    vaeWeights: import('./vaeDecoder').VAEDecoderWeights;
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
    private isLoaded;
    constructor();
    /**
     * GGUF 모델 헤더를 로드하고 가중치 오프셋 테이블을 구축합니다 (Zero WASM Heap).
     */
    loadModel(headerBuffer: ArrayBuffer): Promise<GGUFHeader>;
    /**
     * 텍스트 프롬프트로부터 고해상도 이미지를 생성하는 완전한 E2E 파이프라인 실행 함수.
     */
    generate(options: GenerationOptions): Promise<DecodedImage>;
}
