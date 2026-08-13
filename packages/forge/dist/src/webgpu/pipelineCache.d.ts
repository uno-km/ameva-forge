/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * pipelineCache.ts — WGSL 컴파일 파이프라인 캐시
 *
 * L-03 Fix: clear() 메서드를 통해 device lost 시 캐시 무효화.
 * NL-02 Fix: 캐시 키에 WGSL 해시를 포함하여 동일 op명으로 다른 WGSL 지원.
 */
/**
 * WHAT: 문자열 데이터를 기반으로 고유한 해시(32-bit 정수형 기반 16진수 문자열)를 생성하는 간단한 해시 함수입니다.
 * WHY: WGSL 소스 코드 문자열을 해시화하여 파이프라인 캐시 키(cache key)에 추가함으로써, 같은 이름의 연산(op)이라도 다른 WGSL 코드가 주어질 경우 충돌을 방지하기 위함입니다.
 * HOW: djb2 해시 알고리즘 변형을 사용합니다.
 *      초기값 5381에서 시작하여 문자열의 각 문자 코드를 순회(for 루프)하면서,
 *      비트 이동 및 XOR 연산을 통해 해시 값을 누적한 후 부호 없는 32비트 정수를 16진수 문자열로 변환하여 반환합니다.
 */
export declare function hashString(str: string): string;
/**
 * WHAT: WebGPU 컴퓨트 파이프라인(GPUComputePipeline)과 셰이더 모듈(GPUShaderModule)을 저장하고 재사용하는 캐시 관리 클래스입니다.
 * WHY: WGSL 코드를 매번 파싱하고 컴파일하는 비용(오버헤드)을 줄여 GPU 연산 초기화 성능을 극대화하기 위해 존재합니다.
 * HOW: 내부적으로 Map 인스턴스를 유지하여 연산명(key)과 WGSL 해시의 조합을 캐시 키로 사용하고, 컴파일된 객체를 메모리에 저장 및 반환합니다.
 */
declare class PipelineCache {
    /**
     * WHAT: 컴파일 완료된 셰이더 모듈과 컴퓨트 파이프라인 객체를 키(문자열)에 매핑하여 보관하는 내부 저장소입니다.
     * WHY: 반복적인 연산 요청 시 동일한 코드가 주어지면 이전에 컴파일된 객체를 빠르게 찾아 반환하기 위해 필요합니다.
     * HOW: JavaScript 내장 Map 구조를 사용하여 생성되며, 캐시 적중(Cache Hit) 시 저장된 값을 제공하고, 누락(Cache Miss) 시 새 객체를 추가합니다.
     */
    private cache;
    /**
     * WHAT: 주어진 연산 이름(key)과 WGSL 소스 코드를 바탕으로 컴파일된 파이프라인 객체와 셰이더 모듈을 반환합니다.
     * WHY: 기존에 컴파일된 캐시가 있다면 즉시 반환하여 성능을 최적화하고, 없다면 즉석에서(Synchronously) 새로 컴파일하기 위해 사용됩니다.
     * HOW: 연산 이름과 WGSL 해시를 조합해 캐시 키를 만든 후, 내부 캐시 맵에서 조회합니다. 없을 경우 WebGPU 디바이스에 셰이더 모듈과 파이프라인을 생성 요청하고, 결과를 캐시에 저장한 뒤 반환합니다.
     */
    getPipeline(key: string, wgslCode: string): {
        shader: GPUShaderModule;
        pipeline: GPUComputePipeline;
    };
    /**
     * WHAT: 애플리케이션 초기화 단계에서 지정된 여러 파이프라인을 비동기적으로 미리 컴파일하여 캐싱하는 웜업(Warmup) 기능입니다.
     * WHY: 첫 번째 GPU 연산 실행 시 발생하는 동기적 컴파일로 인한 UI 프리징(Freeze) 혹은 끊김(Stuttering) 현상을 방지하기 위해 존재합니다.
     * HOW: 입력받은 배열을 순회하며 아직 캐시되지 않은 항목만 추려낸 뒤, `createComputePipelineAsync`를 사용해 비동기로 병렬 컴파일을 수행(Promise.allSettled)하고 결과를 캐시에 저장합니다.
     */
    warmup(entries: Array<{
        key: string;
        wgslCode: string;
    }>): Promise<void>;
    /**
     * WHAT: 파이프라인 캐시 내에 저장된 모든 항목을 삭제하여 완전히 초기화합니다.
     * WHY: GPU 디바이스가 유실(device lost)되었거나 초기화가 다시 발생할 때, 이전 디바이스 컨텍스트를 가리키는 더 이상 유효하지 않은 파이프라인 참조를 제거하기 위함입니다.
     * HOW: 내부 맵(Map) 객체의 내장 메서드인 `clear()`를 호출하여 모든 키-값 쌍을 비웁니다.
     */
    clear(): void;
    /**
     * WHAT: 현재 캐시에 저장된 파이프라인 객체의 총 개수를 반환하는 프로퍼티 접근자(Getter)입니다.
     * WHY: 메모리 사용량 모니터링이나 디버깅 시 캐시의 누적 상태를 파악하기 위해 제공됩니다.
     * HOW: 내부 캐시 맵의 `size` 프로퍼티 값을 그대로 반환합니다.
     */
    get size(): number;
}
/**
 * WHAT: 전역에서 공유되는 단일 PipelineCache 인스턴스입니다.
 * WHY: 여러 텐서 연산 모듈이 개별 캐시를 만들지 않고 하나의 중앙 집중형 캐시를 재사용하여 메모리와 컴파일 비용을 최소화하기 위해 사용됩니다.
 * HOW: PipelineCache 클래스의 인스턴스를 하나 생성하여 모듈 외부로 노출(export)합니다.
 */
export declare const _globalPipelineCache: PipelineCache;
export {};
