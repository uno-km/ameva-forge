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
/**
 * WHAT: 주어진 입력 데이터를 검증하고 안전하게 Float32Array 형태로 변환 혹은 반환하는 함수입니다. (H-05 Fix 적용)
 * WHY: 데이터의 중복 복사를 막아(Zero-Copy) 대용량 텐서 데이터 전송 시의 성능 저하를 방지하면서도, 데이터 타입의 일관성을 유지하기 위해 존재합니다.
 * HOW: 이미 Float32Array 형태이면 원본 그대로 반환합니다. PyProxy인 경우 toJs() 결과를 추출한 후 Float32Array이면 그대로 반환하고, ArrayBuffer라면 새로운 Float32Array 뷰로 래핑하여 반환합니다. 그 외에는 예외를 던집니다.
 *
 * @param input Pyodide Proxy 객체이거나 ArrayBuffer/Float32Array 형태의 데이터
 * @returns 확보된 Float32Array
 */
export declare function ensureFloat32Array(input: unknown): Float32Array;
/**
 * WHAT: 입력 데이터를 강제로 새로운 메모리 공간에 깊은 복사(Deep Copy)하여 반환하는 함수입니다. (명시적 deep copy 용도)
 * WHY: 원본 데이터(WASM 힙 등)가 삭제되거나 변경되어도 안전하게 데이터를 보존해야 할 때, 혹은 독립적인 데이터 소유권을 가지는 버퍼가 필요할 때 호출하기 위해 존재합니다. 일반 데이터 읽기에는 성능상 사용하지 않아야 합니다.
 * HOW: ensureFloat32Array를 호출하여 먼저 안전한 뷰를 확보한 뒤, new Float32Array(view)를 사용하여 동일한 요소들을 가지는 완전히 새로운 메모리 배열 인스턴스를 할당하여 반환합니다.
 *
 * @param input 원본 데이터
 * @returns 독립된 메모리 공간을 가지는 복사된 Float32Array
 */
export declare function cloneToFloat32Array(input: unknown): Float32Array;
