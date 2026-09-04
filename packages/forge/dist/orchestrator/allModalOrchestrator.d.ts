/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-334 Grand Unified All-Modal On-Device AI Orchestrator
 *
 * WHAT: STT(귀), LLM(뇌), Vision(눈), TTS(입), Diffusion(손)을 단일 온디바이스 텐서 런타임으로
 *      조율하는 그랜드 유니파이드 올모달(All-Modal) 오케스트레이터입니다.
 * WHY: 5대 멀티모달 영역의 분편화를 종식시키고, 100% 클라이언트 온디바이스(Zero Cloud Egress) 환경에서
 *      침묵 폴백 없는 엄격한 수치 연산과 비동기 협업 파이프라인을 단일 진입점으로 제공하기 위해 존재합니다.
 * HOW: STTEngine + LLMEngine + CLIPVisionEncoder/ClassicalCV + TTSEngine + WebGPUDiffusionPipeline.
 */
import { CLIPVisionWeights } from '../vision/clipVisionEncoder';
import { LLMWeights, KVCache } from '../llm/llmEngine';
import { WebGPUDiffusionPipeline, GenerationOptions } from '../diffusion/pipeline';
import { DecodedImage } from '../diffusion/vaeDecoder';
export interface AllModalCapabilities {
    readonly modalities: readonly ['stt', 'llm', 'vision', 'tts', 'diffusion'];
    readonly zero_silent_fallback_enforced: boolean;
    readonly on_device_runtime: string;
}
export declare const ALL_MODAL_CAPABILITIES: AllModalCapabilities;
export declare class AllModalOrchestrator {
    diffusionPipeline: WebGPUDiffusionPipeline;
    constructor();
    /**
     * 1. STT (귀): 16kHz PCM 오디오를 80채널 로그 멜-스펙트로그램으로 변환하여 분석합니다.
     */
    listen(pcm: Float32Array, sampleRate?: number): {
        mels: Float32Array;
        numFrames: number;
    };
    /**
     * 2. LLM (뇌): RoPE, RMSNorm, SwiGLU 기반 트랜스포머 디코더로 토큰 로짓을 예측합니다.
     */
    think(tokenId: number, pos: number, weights: LLMWeights, kvCaches: KVCache[]): Float32Array;
    /**
     * 3. Vision (눈): RGBA 이미지로부터 에지를 검출하거나 ViT를 통해 768차원 시맨틱 특징 벡터를 추출합니다.
     */
    seeEdges(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): Uint8Array;
    seeEmbeddings(rgb: Float32Array, width: number, height: number, weights: CLIPVisionWeights): Float32Array;
    /**
     * 4. TTS (입): 로젠버그 성문 펄스와 5-밴드 바이쿼드 필터로 실시간 PCM 오디오를 합성합니다.
     */
    speak(text: string, sampleRate?: number): {
        pcm: Float32Array;
        durationSeconds: number;
    };
    /**
     * 5. Diffusion (손): 텍스트 프롬프트로부터 온디바이스 신경망 디퓨전 파이프라인으로 이미지를 그립니다.
     */
    draw(options: GenerationOptions): Promise<DecodedImage>;
}
