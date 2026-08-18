/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
 *
 * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
 *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
 *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
 */
export type SafeCopyOptions = {
    retryDetached?: boolean;
    reacquire?: () => Float32Array;
};
export declare function ensureFloat32Array(input: unknown, options?: SafeCopyOptions): Float32Array;
/**
 * WHAT: 입력 데이터를 강제로 새로운 메모리 공간에 깊은 복사(Deep Copy)하여 반환하는 함수입니다. (명시적 deep copy 용도)
 * WHY: 원본 데이터(WASM 힙 등)가 삭제되거나 변경되어도 안전하게 데이터를 보존해야 할 때, 혹은 독립적인 데이터 소유권을 가지는 버퍼가 필요할 때 호출하기 위해 존재합니다. 일반 데이터 읽기에는 성능상 사용하지 않아야 합니다.
 * HOW: ensureFloat32Array를 호출하여 먼저 안전한 뷰를 확보한 뒤, new Float32Array(view)를 사용하여 동일한 요소들을 가지는 완전히 새로운 메모리 배열 인스턴스를 할당하여 반환합니다.
 *
 * @param input 원본 데이터
 * @returns 독립된 메모리 공간을 가지는 복사된 Float32Array
 */
export declare function cloneToFloat32Array(input: unknown): Float32Array;
