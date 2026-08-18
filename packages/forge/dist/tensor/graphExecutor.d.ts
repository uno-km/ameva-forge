/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * graphExecutor.ts — JSON 그래프 파서 & GPU 스케줄러
 *
 * C-04 Fix: JSON 입력에 대한 강력한 검증 추가
 * M-05 Fix: matmul dispatch X/Y swap 수정
 * H-01 Fix: _globalPipelineCache를 모든 op에 적용
 * NC-06 Fix: inst.in null-guard 추가 (! 비null 단언 제거)
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * NM-05 Fix: device.pushErrorScope()로 op별 에러 감지
 */
import { TensorRegistry } from "./tensorRegistry";
import { TensorHandle, DType } from "../types";
import { AllocationToken } from "../webgpu/quota";
export type ForgeRuntimeConfig = {
    workloadBudgetElements?: number;
    maxOpsPerSubmit?: number;
    maxShapeDim?: number;
    maxElements?: number;
    maxInstructions?: number;
    allowNonFinite?: boolean;
};
export declare function configureRuntime(config: ForgeRuntimeConfig): void;
export declare function getRuntimeConfig(): Required<ForgeRuntimeConfig>;
export interface PendingTensorRecord {
    handle: TensorHandle;
    buffer: GPUBuffer;
    token: AllocationToken;
    shape: number[];
    dtype: DType;
    byteLength: number;
}
/**
 * WHAT: 롤백 과정에서 즉시 destroy에 실패한 GPU 버퍼들의 지연 해제를 재시도합니다.
 * WHY: 일시적 GPU busy 상태 등으로 파괴 실패 시 유령 VRAM 누수를 방지합니다.
 */
export declare function processDeferredGC(): void;
export declare class GraphTransaction {
    private readonly pending;
    add(record: PendingTensorRecord): void;
    get(handle: TensorHandle): PendingTensorRecord | undefined;
    get handles(): TensorHandle[];
    commit(registry: TensorRegistry): void;
    rollback(): void;
}
export declare function executeGraph(instructionsJson: string, inputs: (Float32Array | any)[], outputIds?: number[]): Promise<Record<string, TensorHandle>>;
