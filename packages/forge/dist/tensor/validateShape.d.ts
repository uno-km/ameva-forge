import { DType } from "../types";
/**
 * validateShape — 텐서 shape의 유효성을 검증하고 총 원소 수를 반환한다.
 *
 * M-01 Fix: dtype별 바이트 크기를 BYTES_PER_ELEMENT 맵으로 정확히 계산.
 * NM-06 Fix: rank 0 스칼라 텐서 허용 (PyTorch/JAX/TF 표준).
 *   rank 0 = shape=[], elements=1, byteLength=4 (단일 float32 스칼라)
 */
export declare function validateShape(shape: number[], dtype: DType, expectedByteLength?: number): number;
