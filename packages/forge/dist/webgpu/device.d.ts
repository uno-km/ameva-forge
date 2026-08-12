/**
 * device.ts — WebGPU 싱글톤 디바이스 래퍼
 *
 * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
 * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
 */
export declare function initWebGPU(options?: GPURequestAdapterOptions): Promise<void>;
export declare function getDevice(): GPUDevice;
/** H-04: adapter.limits 접근용 */
export declare function getAdapter(): GPUAdapter | null;
export declare function getQueue(): GPUQueue;
export declare function isAvailable(): boolean;
export declare function setDeviceLostCallback(callback: () => void): void;
