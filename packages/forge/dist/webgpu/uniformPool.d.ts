import { AllocationToken } from "./quota";
export type UniformEntry = {
    buffer: GPUBuffer;
    token: AllocationToken;
    byteLength: number;
    inFlight: boolean;
    fenceId: number;
};
export declare class UniformBufferPool {
    private pools;
    private inFlight;
    private fenceCounter;
    acquire(byteLength: number): UniformEntry;
    releaseAfterSubmit(entry: UniformEntry): void;
    releaseSync(entry: UniformEntry): void;
    inFlightBytes(): number;
    retireSubmitted(device: GPUDevice): Promise<void>;
    clear(): void;
    private bucket;
}
export declare const _globalUniformPool: UniformBufferPool;
