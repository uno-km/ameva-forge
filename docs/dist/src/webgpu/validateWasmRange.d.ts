/**
 * WHAT: 주어진 오프셋과 데이터 길이가 WASM 선형 메모리 힙(heap)의 유효한 범위 내에 있는지 안전하게 검사합니다.
 * WHY: CPU-GPU 간 데이터 전송이나 공유 메모리 접근 시 악의적이거나 잘못된 크기 요청으로 인한 메모리 침범을 방어하기 위해 호출됩니다.
 * HOW: `Number.isSafeInteger`와 비음수(non-negative) 조건을 통해 입력 인자의 데이터 타입을 엄격히 검증한 후, `offset + byteLength`가 총 WASM 메모리 크기를 초과하지 않는지 계산하여 확인합니다. 위반 시 보안 예외를 던집니다.
 */
export declare function assertWasmRange(offset: number, byteLength: number, wasmByteLength: number): void;
