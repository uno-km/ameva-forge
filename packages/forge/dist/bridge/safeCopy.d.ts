/**
 * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
 *
 * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
 *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
 *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
 */
/**
 * H-05 Fix: 입력이 이미 Float32Array면 복사 없이 반환.
 * PyProxy인 경우 toJs() 결과를 확인하고 역시 복사 없이 뷰를 반환.
 */
export declare function ensureFloat32Array(input: unknown): Float32Array;
/**
 * 명시적 deep copy가 필요한 경우 (예: 버퍼 소유권 이전).
 * 일반 데이터 읽기에는 사용하지 말 것.
 */
export declare function cloneToFloat32Array(input: unknown): Float32Array;
