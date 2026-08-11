/**
 * tensorRegistry.ts — GPU 텐서 생명주기 레지스트리
 *
 * C-06 Fix: dispose() 시 _globalQuotaManager.markPendingRelease() 즉시 호출.
 * NC-07 Fix: dynamic import() 제거 → 정적 import 사용 + device.destroy() 보장.
 * NL-03 Fix: Date.now() 제거 → 단조증가 ID만 사용 (타이밍 정보 노출 방지).
 */
import { TensorHandle, TensorRecord } from "../types";
declare class TensorRegistry {
    private records;
    private nextId;
    register(recordOmitHandle: Omit<TensorRecord, 'handle' | 'disposed' | 'createdAt'>): TensorHandle;
    get(handle: TensorHandle): TensorRecord;
    has(handle: TensorHandle): boolean;
    dispose(handle: TensorHandle): void;
    clear(): void;
}
export declare const _globalRegistry: TensorRegistry;
export {};
