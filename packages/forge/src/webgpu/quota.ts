/**
 * quota.ts — VRAM 할당 쿼터 관리자
 *
 * C-06 Fix: quota release 타이밍 불일치 해결 — markPendingRelease + release 2단계.
 * H-04 Fix: setLimits()로 런타임에 동적 쿼터 설정 가능.
 * NH-04 Fix: markPendingRelease() 이중 dispose 시 카운터 음수 방지.
 */

import { AMEVAForgeQuotaExceededError } from "../errors";

export class QuotaManager {
  /** 현재 할당된 총 바이트 (pending 포함) */
  public allocatedBytes: number = 0;
  /** GPU 큐 대기 중인 해제 바이트 (실제로는 아직 GPU 점유 중) */
  public pendingReleaseBytes: number = 0;

  public hardLimitBytes: number;
  public softLimitBytes: number;

  constructor(
    hardLimitBytes: number = 1 * 1024 * 1024 * 1024,  // H-NEW-02: 기본 1GB (보수적)
    softLimitBytes: number = 768 * 1024 * 1024         // 768MB
  ) {
    this.hardLimitBytes = hardLimitBytes;
    this.softLimitBytes = softLimitBytes;
  }

  /** H-04: 런타임에 동적으로 쿼터 상한 재설정 */
  setLimits(hardLimitBytes: number, softLimitBytes: number): void {
    if (!Number.isSafeInteger(hardLimitBytes) || hardLimitBytes <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid hard limit: ${hardLimitBytes}`);
    }
    if (!Number.isSafeInteger(softLimitBytes) || softLimitBytes <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid soft limit: ${softLimitBytes}`);
    }
    if (softLimitBytes > hardLimitBytes) {
      throw new AMEVAForgeQuotaExceededError("softLimitBytes must be <= hardLimitBytes");
    }
    this.hardLimitBytes = hardLimitBytes;
    this.softLimitBytes = softLimitBytes;
  }

  reserve(byteLength: number): void {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      throw new AMEVAForgeQuotaExceededError(`Invalid allocation size: ${byteLength}`);
    }
    // C-06: 여유 공간 = hardLimit - allocatedBytes (pending 포함, 보수적)
    if (byteLength > this.hardLimitBytes - this.allocatedBytes) {
      throw new AMEVAForgeQuotaExceededError(
        `Quota Exceeded: Cannot allocate ${byteLength} bytes. ` +
        `Current: ${this.allocatedBytes} (${this.pendingReleaseBytes} pending release), ` +
        `Limit: ${this.hardLimitBytes}`
      );
    }
    this.allocatedBytes += byteLength;
    if (this.allocatedBytes - this.pendingReleaseBytes > this.softLimitBytes) {
      console.warn(
        `[AMEVA] VRAM soft quota exceeded: ` +
        `${((this.allocatedBytes - this.pendingReleaseBytes) / 1e9).toFixed(2)}GB / ` +
        `${(this.softLimitBytes / 1e9).toFixed(2)}GB`
      );
    }
  }

  track(byteLength: number): void {
    this.reserve(byteLength);
  }

  /**
   * C-06: dispose() 호출 시 즉시 "해제 예정"으로 표시.
   * NH-04 Fix: 이중 dispose 방지 — allocatedBytes 기준으로 클램핑하되
   *   pendingReleaseBytes가 allocatedBytes를 초과하지 않도록 보장.
   */
  markPendingRelease(byteLength: number): void {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) return;
    // NH-04: 이미 pending으로 마킹된 bytes + 새로운 bytes가 allocated를 초과하지 않도록
    const newPending = this.pendingReleaseBytes + byteLength;
    this.pendingReleaseBytes = Math.min(newPending, this.allocatedBytes);
  }

  /**
   * GPU 큐 완료 후 실제 해제 확정.
   * NH-04 Fix: 음수 방지를 위해 Math.max(0, ...) 적용.
   */
  release(byteLength: number): void {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) return;
    this.allocatedBytes = Math.max(0, this.allocatedBytes - byteLength);
    this.pendingReleaseBytes = Math.max(0, this.pendingReleaseBytes - byteLength);
  }

  getUsage(): {
    allocatedBytes: number;
    pendingReleaseBytes: number;
    effectiveBytes: number;
    hardLimitBytes: number;
    softLimitBytes: number;
  } {
    return {
      allocatedBytes: this.allocatedBytes,
      pendingReleaseBytes: this.pendingReleaseBytes,
      effectiveBytes: this.allocatedBytes - this.pendingReleaseBytes,
      hardLimitBytes: this.hardLimitBytes,
      softLimitBytes: this.softLimitBytes
    };
  }

  reset(): void {
    this.allocatedBytes = 0;
    this.pendingReleaseBytes = 0;
  }
}

export const _globalQuotaManager = new QuotaManager();
