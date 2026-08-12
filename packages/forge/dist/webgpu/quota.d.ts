/**
 * quota.ts — VRAM 할당 쿼터 관리자
 *
 * C-06 Fix: quota release 타이밍 불일치 해결 — markPendingRelease + release 2단계.
 * H-04 Fix: setLimits()로 런타임에 동적 쿼터 설정 가능.
 * NH-04 Fix: markPendingRelease() 이중 dispose 시 카운터 음수 방지.
 */
export declare class QuotaManager {
    /** 현재 할당된 총 바이트 (pending 포함) */
    allocatedBytes: number;
    /** GPU 큐 대기 중인 해제 바이트 (실제로는 아직 GPU 점유 중) */
    pendingReleaseBytes: number;
    hardLimitBytes: number;
    softLimitBytes: number;
    constructor(hardLimitBytes?: number, // H-NEW-02: 기본 1GB (보수적)
    softLimitBytes?: number);
    /** H-04: 런타임에 동적으로 쿼터 상한 재설정 */
    setLimits(hardLimitBytes: number, softLimitBytes: number): void;
    reserve(byteLength: number): void;
    /**
     * C-06: dispose() 호출 시 즉시 "해제 예정"으로 표시.
     * NH-04 Fix: 이중 dispose 방지 — allocatedBytes 기준으로 클램핑하되
     *   pendingReleaseBytes가 allocatedBytes를 초과하지 않도록 보장.
     */
    markPendingRelease(byteLength: number): void;
    /**
     * GPU 큐 완료 후 실제 해제 확정.
     * NH-04 Fix: 음수 방지를 위해 Math.max(0, ...) 적용.
     */
    release(byteLength: number): void;
    getUsage(): {
        allocatedBytes: number;
        pendingReleaseBytes: number;
        effectiveBytes: number;
        hardLimitBytes: number;
        softLimitBytes: number;
    };
    reset(): void;
}
export declare const _globalQuotaManager: QuotaManager;
