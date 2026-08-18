/**
 * dispatchShape.ts - 2D WebGPU Workgroup Dispatch Calculator
 * 
 * WHAT: 1D 요소 수(numElements)를 WebGPU의 2D 디스패치 그리드(dispatchX, dispatchY)로 안전하게 분할하는 공용 유틸리티입니다.
 * WHY: WebGPU 디스패치 차원당 한도(65,535)를 초과하는 대용량 텐서(> 4.19M 원소)에서 연산이 절단되는 Silent Truncation 버그를 원천 차단합니다.
 * HOW: dispatchX = min(totalWorkgroups, maxPerDim), dispatchY = ceil(totalWorkgroups / maxPerDim)로 2D 그리드를 계산합니다.
 */
import { AMEVAForgeValidationError } from "../errors";

export type Dispatch2D = {
  dispatchX: number;
  dispatchY: number;
  workgroupsX: number;
  totalWorkgroups: number;
};

export function computeDispatch2D(
  numElements: number,
  workgroupSize: number = 64,
  maxPerDim: number = 65535
): Dispatch2D {
  if (!Number.isSafeInteger(numElements) || numElements <= 0) {
    throw new AMEVAForgeValidationError(`Invalid numElements: ${numElements}`);
  }

  const totalWorkgroups = Math.ceil(numElements / workgroupSize);
  const dispatchX = Math.min(totalWorkgroups, maxPerDim);
  const dispatchY = Math.ceil(totalWorkgroups / maxPerDim);

  if (dispatchY > maxPerDim) {
    throw new AMEVAForgeValidationError(
      `Dispatch too large: ${totalWorkgroups} workgroups exceeds 2D WebGPU limit (${maxPerDim}x${maxPerDim})`
    );
  }

  return {
    dispatchX,
    dispatchY,
    workgroupsX: dispatchX,
    totalWorkgroups,
  };
}
