/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-345 High-Level Plug & Play GGUF Model Loader & Session Engine
 *
 * WHAT: Hugging Face URL, 브라우저 File 드래그 앤 드롭, 또는 ArrayBuffer로부터 GGUF 모델을 로드하고,
 *      브라우저 로컬 캐시(CacheStorage/OPFS), HTTP Range-Request 스트리밍, 메타데이터 파싱,
 *      VRAM Direct DMA 가중치 주입, 그리고 토크나이저 결합까지 1-클릭으로 완결 짓는 고수준 모델 로더입니다.
 * WHY: 개발자가 저수준 셰이더나 텐서 버퍼를 일일이 다루지 않고도, 외부 GGUF 모델을 즉각 브라우저에
 *      '끼워 넣기(Plug & Play)'만 하면 동작하는 최고급 온디바이스 AI 런타임 경험을 제공하기 위함입니다.
 * HOW: Cache Lookup -> Stream Buffer -> GGUFStreamer.parseHeader -> BPETokenizer Binding -> LLMTextGenerator Wrap.
 */
import { GGUFHeader } from './ggufStreamer';
import { BPETokenizer } from '../tokenizer/bpeTokenizer';
import { LLMTextGenerator, TextGenerationOptions } from '../llm/llmTextGenerator';
import { LLMWeights } from '../llm/llmEngine';
export interface ModelLoadProgress {
    stage: 'checking_cache' | 'downloading' | 'parsing_header' | 'loading_tensors' | 'ready';
    loadedBytes: number;
    totalBytes: number;
    percentage: number;
    statusText: string;
}
export interface ModelLoadOptions {
    onProgress?: (progress: ModelLoadProgress) => void;
    cacheKey?: string;
    useCache?: boolean;
    backend?: 'webgpu' | 'cpu';
}
export interface ModelPresetInfo {
    id: string;
    name: string;
    architecture: string;
    parameterCount: string;
    quantization: string;
    url: string;
    fileSizeMb: number;
}
/**
 * 널리 검증된 초경량 온디바이스 추천 모델 프리셋 카탈로그 (Hugging Face Direct CDN)
 */
export declare const OFFICIAL_MODEL_PRESETS: readonly ModelPresetInfo[];
export declare class LoadedModelSession {
    readonly header: GGUFHeader;
    readonly tokenizer: BPETokenizer;
    readonly generator: LLMTextGenerator;
    readonly architecture: string;
    readonly weights?: LLMWeights;
    constructor(header: GGUFHeader, tokenizer: BPETokenizer, generator: LLMTextGenerator, weights?: LLMWeights);
    /**
     * 자연어 프롬프트를 주입하여 답변을 실시간 스트리밍으로 생성합니다.
     */
    prompt(text: string, options?: TextGenerationOptions): Promise<string>;
}
export declare class ModelLoader {
    private static readonly CACHE_NAME;
    /**
     * 다양한 소스(URL, File, ArrayBuffer)로부터 GGUF 모델을 로드하여 세션을 생성합니다.
     */
    static loadModel(source: string | File | ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModelSession>;
    private static fetchWithCache;
    private static readFile;
}
