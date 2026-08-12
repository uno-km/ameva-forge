import { AMEVAForgeShapeError, AMEVAForgeDTypeError } from "../errors";
import { DType } from "../types";

/** dtype별 바이트 크기 */
const BYTES_PER_ELEMENT: Record<DType, number> = {
  "float32": 4,
  // float16: 2 — 셰이더 구현 완료 후 추가 예정
  // int32: 4 — 셰이더 구현 완료 후 추가 예정
};

const MAX_ELEMENTS = 256 * 1024 * 1024;
const MAX_RANK = 8; // NM-06: 스칼라(rank 0) 포함하여 0~8까지 허용

/**
 * validateShape — 텐서 shape의 유효성을 검증하고 총 원소 수를 반환한다.
 *
 * M-01 Fix: dtype별 바이트 크기를 BYTES_PER_ELEMENT 맵으로 정확히 계산.
 * NM-06 Fix: rank 0 스칼라 텐서 허용 (PyTorch/JAX/TF 표준).
 *   rank 0 = shape=[], elements=1, byteLength=4 (단일 float32 스칼라)
 */
export function validateShape(
  shape: number[],
  dtype: DType,
  expectedByteLength?: number
): number {
  if (!Array.isArray(shape)) {
    throw new AMEVAForgeShapeError("Shape must be an array.");
  }
  // NM-06 Fix: rank 0 (shape=[]) 허용 — 스칼라 텐서
  if (shape.length > MAX_RANK) {
    throw new AMEVAForgeShapeError(
      `Shape rank must be between 0 and ${MAX_RANK}, got ${shape.length}.`
    );
  }

  let elements = 1;
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    if (!Number.isSafeInteger(dim) || dim <= 0) {
      throw new AMEVAForgeShapeError(`shape[${i}] must be positive, got ${dim}`);
    }
    if (dim > Number.MAX_SAFE_INTEGER / elements) {
      throw new AMEVAForgeShapeError("Shape product overflows safe integer limit.");
    }
    elements *= dim;
  }

  if (elements > MAX_ELEMENTS) {
    throw new AMEVAForgeShapeError(
      `Tensor size exceeds max elements limit: ${elements} > ${MAX_ELEMENTS}`
    );
  }

  if (expectedByteLength !== undefined) {
    const bytesPerElement = BYTES_PER_ELEMENT[dtype];
    if (bytesPerElement === undefined) {
      throw new AMEVAForgeDTypeError(
        `Unsupported dtype for byte size calculation: "${dtype}". ` +
        `Supported: ${Object.keys(BYTES_PER_ELEMENT).join(', ')}`
      );
    }
    const calculatedBytes = elements * bytesPerElement;
    if (calculatedBytes !== expectedByteLength) {
      throw new AMEVAForgeShapeError(
        `Shape/data size mismatch: shape ${JSON.stringify(shape)} (${dtype}) ` +
        `implies ${calculatedBytes} bytes, but data is ${expectedByteLength} bytes.`
      );
    }
  }

  return elements;
}
