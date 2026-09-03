import { DType } from "../types";
/**
 * WHAT: 텐서 shape의 유효성을 검증하고 총 원소 수를 반환하는 함수입니다.
 * WHY: 잘못된 텐서 형태나 예상치 못한 크기의 메모리 할당을 사전에 차단하여 안전한 연산을 보장하기 위함입니다.
 * HOW: 입력된 shape가 배열인지, 랭크 제한을 넘지 않는지 확인한 후, 각 차원의 값을 곱해 총 원소 수를 구합니다. 예상 바이트 크기가 주어진 경우 이를 함께 검증합니다.
 *
 * M-01 Fix: dtype별 바이트 크기를 BYTES_PER_ELEMENT 맵으로 정확히 계산.
 * NM-06 Fix: rank 0 스칼라 텐서 허용 (PyTorch/JAX/TF 표준).
 *   rank 0 = shape=[], elements=1, byteLength=4 (단일 float32 스칼라)
 */
export declare function validateShape(shape: number[], dtype: DType, expectedByteLength?: number): number;
