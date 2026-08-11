/**
 * tensorRegistry.ts — GPU 텐서 생명주기 레지스트리
 *
 * C-06 Fix: dispose() 시 _globalQuotaManager.markPendingRelease() 즉시 호출.
 * NC-07 Fix: dynamic import() 제거 → 정적 import 사용 + device.destroy() 보장.
 * NL-03 Fix: Date.now() 제거 → 단조증가 ID만 사용 (타이밍 정보 노출 방지).
 */

import { TensorHandle, TensorRecord } from "../types";
import { AMEVATensorDisposedError } from "../errors";
import { freeBuffer } from "../webgpu/buffers";
import { _globalQuotaManager } from "../webgpu/quota";
import { getDevice } from "../webgpu/device";

class TensorRegistry {
  private records: Map<TensorHandle, TensorRecord> = new Map();
  private nextId: number = 1;

  register(
    recordOmitHandle: Omit<TensorRecord, 'handle' | 'disposed' | 'createdAt'>
  ): TensorHandle {
    // NL-03 Fix: Date.now() 제거, 단순 카운터만 사용
    const handle = `tensor_${this.nextId++}`;
    const record: TensorRecord = {
      ...recordOmitHandle,
      handle,
      disposed: false,
      createdAt: this.nextId - 1  // NL-03 Fix: monotonic counter, not timestamp
    };
    this.records.set(handle, record);
    return handle;
  }

  get(handle: TensorHandle): TensorRecord {
    const record = this.records.get(handle);
    if (!record) {
      throw new AMEVATensorDisposedError(`Tensor not found: ${handle}`);
    }
    if (record.disposed) {
      throw new AMEVATensorDisposedError(`Attempted to access disposed tensor: ${handle}`);
    }
    return record;
  }

  has(handle: TensorHandle): boolean {
    const record = this.records.get(handle);
    return record !== undefined && !record.disposed;
  }

  dispose(handle: TensorHandle): void {
    const record = this.records.get(handle);
    if (!record || record.disposed) return;

    record.disposed = true;
    this.records.delete(handle);

    // C-06 Fix: 즉시 "해제 예약" 표시
    _globalQuotaManager.markPendingRelease(record.byteLength);

    // NC-07 Fix: 정적 import된 getDevice() 사용 (dynamic import 제거)
    try {
      const device = getDevice();
      device.queue.onSubmittedWorkDone().then(() => {
        freeBuffer(record.buffer, record.byteLength);
      }).catch(() => {
        // GPU 큐 실패 → 즉시 소각
        _safeDestroyBuffer(record);
      });
    } catch {
      // device가 없거나 lost → 즉시 quota 해제 + buffer 소각
      _safeDestroyBuffer(record);
    }
  }

  clear(): void {
    const recordsToFree = Array.from(this.records.values()).filter(r => !r.disposed);
    this.records.clear();

    if (recordsToFree.length === 0) return;

    for (const record of recordsToFree) {
      _globalQuotaManager.markPendingRelease(record.byteLength);
    }

    try {
      const device = getDevice();
      device.queue.onSubmittedWorkDone().then(() => {
        for (const record of recordsToFree) {
          freeBuffer(record.buffer, record.byteLength);
        }
      }).catch(() => {
        for (const record of recordsToFree) {
          _safeDestroyBuffer(record);
        }
      });
    } catch {
      // device already lost
      for (const record of recordsToFree) {
        _safeDestroyBuffer(record);
      }
      _globalQuotaManager.reset();
    }
  }
}

/**
 * NC-07 Fix: device lost 상황에서도 buffer.destroy()를 보장하고 quota를 해제.
 */
function _safeDestroyBuffer(record: TensorRecord): void {
  try {
    record.buffer.destroy();
  } catch {
    // buffer가 이미 destroyed
  }
  _globalQuotaManager.release(record.byteLength);
}

export const _globalRegistry = new TensorRegistry();
