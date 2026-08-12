/**
 * pyodideBridge.ts — globalThis.amevaTensor API 등록자
 *
 * H-02 연동: 단일 실행 경로(graphExecutor.ts)로 통합.
 *   executeGraph 시그니처: (instructionsJson: string, jsInputs: unknown) => Record
 *
 * M-06 연동: disposeBatch 추가 (bridge.py의 js_dispose_batch 지원)
 */
import { init, read, dispose, getTensorInfo, mapBufferAsync, readMappedInto, warmupKernels } from "../tensor/gpuCore";
import { executeGraph } from "../tensor/graphExecutor";
import { TensorHandle } from "../types";
export interface AmevaTensorGlobalAPI {
    init: typeof init;
    read: typeof read;
    dispose: typeof dispose;
    getTensorInfo: typeof getTensorInfo;
    mapBufferAsync: typeof mapBufferAsync;
    readMappedInto: typeof readMappedInto;
    executeGraph: typeof executeGraph;
    warmupKernels: typeof warmupKernels;
    /** M-06: batch dispose — handles 배열을 한 번에 해제 */
    disposeBatch: (handles: TensorHandle[]) => void;
}
declare global {
    var amevaTensor: AmevaTensorGlobalAPI | undefined;
}
export declare function registerPyodideBridge(): AmevaTensorGlobalAPI;
