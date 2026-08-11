/**
 * gpuCore.ts — GPU 코어 API (초기화, 텐서 생명주기, 개별 op 디스패치)
 *
 * H-01 Fix: 모든 op에 _globalPipelineCache 적용 (셰이더 재컴파일 방지)
 * NH-03 Fix: quota를 maxStorageBufferBindingSize 기반으로 설정 (maxBufferSize는 단일 버퍼 크기 제한이지 VRAM 용량이 아님)
 * NH-01 Fix: 개별 op 함수들을 internal로 유지, pyodideBridge에서는 executeGraph만 노출
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * ARC-01 Fix: device.pushErrorScope로 OOM 감지 시도
 * L-01 Fix: dispatchKernel 헬퍼로 모든 op의 반복 코드 통합 (DRY)
 */
import { TensorHandle, DType, TensorInfo } from "../types";
/**
 * 커널 레지스트리: 새 커널 추가 시 import 1줄 + 여기 1줄만 추가하면
 * warmupKernels(), graphExecutor 모두 자동으로 반영된다.
 */
export declare const KERNEL_REGISTRY: ReadonlyMap<string, string>;
export declare function resetRuntimeMemory(): void;
export declare function init(options?: GPURequestAdapterOptions & {
    vramLimitBytes?: number;
}): Promise<void>;
/**
 * 모든 커널 파이프라인을 비동기로 사전 컴파일한다.
 * KERNEL_REGISTRY를 동적으로 순회하므로, 새 커널 추가 시 여기를 수정할 필요 없다.
 */
export declare function warmupKernels(): Promise<void>;
export declare function getTensorInfo(handle: TensorHandle): TensorInfo;
export declare function read(handle: TensorHandle): Promise<Float32Array>;
/**
 * C-05 Fix: staging buffer를 핸들 키로 _pendingStagingBuffers에 저장.
 */
export declare function mapBufferAsync(handle: TensorHandle): Promise<void>;
/**
 * C-05 Fix: 핸들 키로 staging buffer를 조회하여 읽기.
 */
export declare function readMappedInto(handle: TensorHandle, outArray: any): void;
export declare function dispose(handle: TensorHandle): void;
export declare function random(shape: number[], dtype?: DType): TensorHandle;
export declare function uploadFloat32Array(data: any, shape: number[]): TensorHandle;
export declare function matmul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
export declare function relu(handle: TensorHandle): TensorHandle;
export declare function add(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
export declare function mul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
export declare function transpose(handle: TensorHandle): TensorHandle;
export declare function relu_backward(handleX: TensorHandle, handleGrad: TensorHandle): TensorHandle;
