/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 */
import { AMEVAForgeShapeError, AMEVAForgeDTypeError } from "../errors";
import { DType } from "../types";

/** 
 * WHAT: dtype별 바이트 크기를 매핑하는 상수 딕셔너리입니다.
 * WHY: 텐서의 전체 바이트 크기를 계산할 때, 데이터 타입마다 차지하는 바이트 수가 다르기 때문에 이를 정확히 계산하기 위해 존재합니다.
 * HOW: Record 유틸리티 타입을 사용하여 DType 문자열을 키로, 바이트 수(number)를 값으로 갖는 객체를 정의합니다.
 */
const BYTES_PER_ELEMENT: Record<DType, number> = {
  "float32": 4,
  // float16: 2 — 셰이더 구현 완료 후 추가 예정
  // int32: 4 — 셰이더 구현 완료 후 추가 예정
};

/**
 * WHAT: 텐서가 가질 수 있는 최대 원소 수를 정의하는 상수입니다.
 * WHY: 메모리 초과(OOM) 오류를 방지하고 시스템의 안정성을 유지하기 위해 하드 리미트를 설정합니다.
 * HOW: 256MB 크기의 float32 버퍼에 맞추어 256 * 1024 * 1024로 값을 할당합니다.
 */
const MAX_ELEMENTS = 256 * 1024 * 1024;

/**
 * WHAT: 텐서 shape의 최대 랭크(차원 수)를 정의하는 상수입니다.
 * WHY: WebGPU에서 처리할 수 있는 차원의 한계를 설정하고, 과도하게 복잡한 다차원 텐서의 생성을 방지합니다.
 * HOW: 스칼라(rank 0)부터 시작하여 최대 8차원까지 허용하도록 숫자 8을 할당합니다.
 */
const MAX_RANK = 8; // NM-06: 스칼라(rank 0) 포함하여 0~8까지 허용

/**
 * WHAT: 텐서 shape의 유효성을 검증하고 총 원소 수를 반환하는 함수입니다.
 * WHY: 잘못된 텐서 형태나 예상치 못한 크기의 메모리 할당을 사전에 차단하여 안전한 연산을 보장하기 위함입니다.
 * HOW: 입력된 shape가 배열인지, 랭크 제한을 넘지 않는지 확인한 후, 각 차원의 값을 곱해 총 원소 수를 구합니다. 예상 바이트 크기가 주어진 경우 이를 함께 검증합니다.
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

  /**
   * WHAT: 텐서의 총 원소 수를 누적하여 저장하는 변수입니다.
   * WHY: shape 배열의 각 차원을 곱하여 텐서 데이터가 차지할 실제 원소의 총 개수를 알아내기 위해 필요합니다.
   * HOW: 스칼라(rank 0)의 경우를 처리하기 위해 1로 초기화됩니다.
   */
  let elements = 1;

  /**
   * WHAT: shape 배열의 각 차원 값을 순회하며 원소 수를 계산하고 유효성을 검사하는 루프입니다.
   * WHY: 모든 차원의 값이 양의 정수인지 확인하고, 안전한 정수 범위를 벗어나는 오버플로우를 감지하기 위해 존재합니다.
   * HOW: 인덱스 i를 0부터 shape.length - 1까지 증가시키며 dim 값을 추출해 검증하고 elements에 누적 곱셈을 수행합니다.
   */
  for (let i = 0; i < shape.length; i++) {
    /**
     * WHAT: 현재 검사 중인 텐서의 특정 차원(dimension)의 크기를 나타내는 변수입니다.
     * WHY: 이 값이 유효한 양의 정수인지 검사하기 위해 루프 내에서 임시로 저장합니다.
     * HOW: shape 배열에서 i번째 인덱스의 값을 참조하여 가져옵니다.
     */
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
    /**
     * WHAT: 입력된 dtype이 차지하는 단일 원소의 바이트 크기를 저장하는 변수입니다.
     * WHY: 전체 텐서의 예상 바이트 크기를 계산하기 위해 요소당 크기를 알아야 합니다.
     * HOW: BYTES_PER_ELEMENT 상수 맵에서 dtype을 키로 사용하여 값을 조회합니다.
     */
    const bytesPerElement = BYTES_PER_ELEMENT[dtype];
    if (bytesPerElement === undefined) {
      throw new AMEVAForgeDTypeError(
        `Unsupported dtype for byte size calculation: "${dtype}". ` +
        `Supported: ${Object.keys(BYTES_PER_ELEMENT).join(', ')}`
      );
    }
    /**
     * WHAT: shape와 dtype을 바탕으로 계산된 텐서의 실제 필요 바이트 크기를 담는 변수입니다.
     * WHY: 사용자가 제시한 expectedByteLength와 비교하여 데이터 정합성을 검증하기 위해 계산합니다.
     * HOW: 누적된 총 원소 수(elements)에 원소당 바이트 크기(bytesPerElement)를 곱하여 구합니다.
     */
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
