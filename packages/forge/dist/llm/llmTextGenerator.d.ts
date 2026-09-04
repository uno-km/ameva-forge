/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-343 Autoregressive LLM Text Streaming Generator
 *
 * WHAT: BPETokenizer, LLMEngine, Sampler를 하나로 유기적으로 결합하여,
 *      자연어 프롬프트로부터 실시간 스트리밍 텍스트 토큰을 생성하고 방출하는 완제품 생성기입니다.
 * WHY: 메인 스레드 락(UI 프리징)과 TDR 크래시를 방지하고, WebGPU 하드웨어 가속을 통해
 *      사용자에게 ChatGPT와 동등한 타자기 효과의 실시간 텍스트 스트리밍 경험을 제공하기 위함입니다.
 * HOW: Tokenize -> Prefill KV-Cache -> Autoregressive Decode Loop (with Event-Loop Yield) -> Sampler -> Decode Chunk.
 */
import { BPETokenizer } from '../tokenizer/bpeTokenizer';
import { LLMWeights } from './llmEngine';
import { SamplingOptions } from './sampler';
export interface TextGenerationOptions extends SamplingOptions {
    maxNewTokens?: number;
    stopTokens?: number[];
    onToken?: (token: string, progress: TextGenerationProgress) => void;
    abortSignal?: AbortSignal;
    backend?: 'webgpu' | 'cpu';
}
export interface TextGenerationProgress {
    tokenCount: number;
    maxTokens: number;
    tps: number;
    elapsedMs: number;
    done: boolean;
}
export declare class LLMTextGenerator {
    tokenizer: BPETokenizer;
    weights?: LLMWeights;
    dim: number;
    vocabSize: number;
    constructor(tokenizer: BPETokenizer, weights?: LLMWeights, dim?: number, vocabSize?: number);
    /**
     * 브라우저 렌더 이벤트 루프에 제어권을 양보하여 UI 멈춤 및 TDR을 방지합니다.
     */
    private static yieldToEventLoop;
    /**
     * 프롬프트 문자열로부터 텍스트를 실시간 스트리밍으로 생성합니다.
     */
    generateStream(prompt: string, options?: TextGenerationOptions): Promise<string>;
}
