/**
 * pipelineCache.ts — WGSL 컴파일 파이프라인 캐시
 *
 * L-03 Fix: clear() 메서드를 통해 device lost 시 캐시 무효화.
 * NL-02 Fix: 캐시 키에 WGSL 해시를 포함하여 동일 op명으로 다른 WGSL 지원.
 */
/**
 * NL-02 Fix: 간단한 문자열 해시 함수 (djb2 변형).
 * 같은 op명으로 다른 WGSL 코드가 전달될 때 캐시 충돌 방지.
 */
export declare function hashString(str: string): string;
declare class PipelineCache {
    private cache;
    /**
     * 주어진 key(op명)와 wgslCode 해시로 캐시를 조회하거나 새로 컴파일하여 반환한다.
     * NL-02 Fix: 캐시 키 = `${key}:${hashString(wgslCode)}`
     */
    getPipeline(key: string, wgslCode: string): {
        shader: GPUShaderModule;
        pipeline: GPUComputePipeline;
    };
    /**
     * H-NEW-08: 비동기 파이프라인 사전 컴파일 (UI freeze 방지).
     * init() 직후 호출하여 첫 연산 시 동기 컴파일 블로킹을 방지한다.
     */
    warmup(entries: Array<{
        key: string;
        wgslCode: string;
    }>): Promise<void>;
    /**
     * L-03 Fix: WebGPU device lost 시 캐시 전체 무효화.
     */
    clear(): void;
    get size(): number;
}
export declare const _globalPipelineCache: PipelineCache;
export {};
