/**
 * types.ts — 핵심 타입 정의
 *
 * H-06 Fix: DType을 실제 지원되는 "float32"만으로 제한.
 *   기존에 float16/int32가 타입에 있었지만 셰이더와 검증 로직이 float32 전용이라
 *   타입 에러 없이 잘못된 셰이더에 전달되는 버그가 있었다.
 *   → float16/int32 추가는 셰이더 커널 구현과 동시에 이루어져야 한다.
 */
export type TensorHandle = string;
/** H-06: 현재 구현이 실제로 지원하는 dtype만 허용 */
export type DType = "float32";
export interface TensorRecord {
    handle: TensorHandle;
    shape: number[];
    dtype: DType;
    byteLength: number;
    buffer: GPUBuffer;
    disposed: boolean;
    /** Monotonic registration order (NOT a timestamp) */
    createdAt: number;
}
export interface TensorInfo {
    handle: TensorHandle;
    shape: number[];
    dtype: DType;
    byteLength: number;
    disposed: boolean;
}
