/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * quota.ts — VRAM 할당 쿼터 관리자
 *
 * C-06 Fix: quota release 타이밍 불일치 해결 — markPendingRelease + release 2단계.
 * H-04 Fix: setLimits()로 런타임에 동적 쿼터 설정 가능.
 * NH-04 Fix: markPendingRelease() 이중 dispose 시 카운터 음수 방지.
 */
/**
 * WHAT: GPU 메모리 할당의 목적이나 종류를 나타내는 리터럴 타입입니다.
 * WHY: 메모리가 어떤 용도로(tensor 데이터, staging 버퍼, uniform 등) 할당되었는지 추적하여 디버깅 및 프로파일링에 활용하기 위해 정의되었습니다.
 * HOW: 문자열 유니온 타입으로 선언되어 'tensor', 'staging', 'uniform', 'temporary' 중 하나의 값을 가집니다.
 */
export type AllocationKind = 'tensor' | 'staging' | 'uniform' | 'temporary';
/**
 * WHAT: 할당된 메모리 토큰의 현재 생명주기 상태를 나타내는 리터럴 타입입니다.
 * WHY: 메모리가 아직 사용 중인지(active), 해제가 예약되었는지(pending_release), 완전히 해제되었는지(released) 구분하여 안전한 리소스 관리를 보장하기 위함입니다.
 * HOW: 상태 전이를 명확하게 나타내기 위해 문자열 유니온을 사용합니다.
 */
export type AllocationState = 'active' | 'pending_release' | 'released';
/**
 * WHAT: 할당된 GPU 메모리 블록을 추적하는 메타데이터 객체(토큰) 클래스입니다.
 * WHY: 실제 GPU 버퍼 리소스와 매핑되어 해당 할당의 크기, 종류, 소유자 등을 식별하고 QuotaManager를 통한 반환을 제어하기 위해 존재합니다.
 * HOW: 생성자에서 유니크 ID, 크기(size), 종류, 그래프 소유자(ownerGraph) 및 생성 세대(generation)를 주입받아 초기화합니다.
 */
export declare class AllocationToken {
    id: string;
    size: number;
    kind: AllocationKind;
    state: AllocationState;
    ownerGraph: string | null;
    generation: number;
    /**
     * WHAT: AllocationToken 인스턴스를 초기화하는 생성자입니다.
     * WHY: 새로운 메모리 할당이 예약될 때 필요한 상태 및 식별 데이터를 객체에 부여하기 위해 호출됩니다.
     * HOW: 전달받은 파라미터로 멤버 변수를 초기화하며, 초기 상태는 'active'로 설정합니다.
     */
    constructor(id: string, size: number, kind: AllocationKind, ownerGraph: string | null, generation: number);
}
/**
 * WHAT: 시스템 전체의 GPU VRAM 할당량을 중앙 집중적으로 관리하는 클래스입니다.
 * WHY: WebGPU 애플리케이션에서 발생할 수 있는 Out-Of-Memory(OOM) 오류를 미연에 방지하고, 메모리 누수를 추적하며 동적으로 한계(Limits)를 설정하기 위해 필요합니다.
 * HOW: 소프트 제한(soft limit)과 하드 제한(hard limit)을 기반으로 메모리 할당 요청을 검사하고 허용하거나 거부하며, 할당된 토큰을 Map을 통해 상태별로 관리합니다.
 */
export declare class QuotaManager {
    /**
     * WHAT: 현재 활성 상태로 할당된 총 메모리 크기(바이트)입니다.
     * WHY: 사용 중인 리소스를 합산하여 쿼터를 초과하지 않는지 감시하기 위해 유지합니다.
     * HOW: reserveToken 시 증가하고, releaseToken 시 감소합니다.
     */
    allocatedBytes: number;
    /**
     * WHAT: 해제가 예약되었으나 아직 완전히 반환되지 않은 메모리 크기(바이트)입니다.
     * WHY: 비동기 작업 중 잠시 유지되는 메모리를 계산하여 여유 한계를 보다 정확하게 산정하기 위함입니다.
     * HOW: markPendingRelease 호출 시 증가하고, 완전히 해제될 때 감소합니다.
     */
    pendingReleaseBytes: number;
    /**
     * WHAT: 메모리 할당이 절대로 초과할 수 없는 최대 허용치(바이트)입니다.
     * WHY: 이 값을 초과하는 할당 요청을 즉시 차단하여 치명적인 시스템 충돌(OOM)을 막기 위해 설정됩니다.
     * HOW: 생성자에서 주입되거나 setLimits를 통해 설정됩니다.
     */
    hardLimitBytes: number;
    /**
     * WHAT: 경고를 발생시키는 메모리 사용량의 임계점(바이트)입니다.
     * WHY: 하드 리밋에 도달하기 전 시스템에 과부하가 올 수 있음을 경고(warn)하기 위해 사용됩니다.
     * HOW: 실제 사용량(allocated - pending)이 이 값을 초과할 때 콘솔에 경고 로그를 출력합니다.
     */
    softLimitBytes: number;
    /**
     * WHAT: 발급된 모든 AllocationToken을 고유 식별자(ID)로 관리하는 맵(Map)입니다.
     * WHY: 토큰의 무결성 검증, 이중 해제(Double Free) 방지 및 전체 할당 현황 조회를 위해 존재합니다.
     * HOW: 토큰 발급 시 추가하고 해제 시 삭제합니다.
     */
    private tokens;
    /**
     * WHAT: 새로 생성되는 메모리 토큰에 부여할 고유 식별자 카운터입니다.
     * WHY: 각 할당 토큰을 구별하고 맵에서 충돌 없이 관리하기 위해 필요합니다.
     * HOW: 새로운 토큰이 생성될 때마다 1씩 증가합니다.
     */
    private nextId;
    /**
     * WHAT: 현재 할당 주기를 나타내는 세대(Generation) 카운터입니다.
     * WHY: 그래프 재컴파일 등 대규모 변경이 일어날 때 이전 세대의 토큰들을 구분하고 메모리 누수를 진단하기 위해 도입되었습니다.
     * HOW: incrementGeneration() 호출 시 증가하며 토큰 생성 시 부여됩니다.
     */
    private currentGeneration;
    /**
     * WHAT: QuotaManager 클래스의 인스턴스를 초기화하는 생성자입니다.
     * WHY: 객체 생성 시 초기 하드 리밋과 소프트 리밋 용량을 설정하기 위해 호출됩니다.
     * HOW: 전달된 바이트 값을 각각의 클래스 프로퍼티에 할당합니다.
     */
    constructor(hardLimitBytes?: number, softLimitBytes?: number);
    /**
     * WHAT: 메모리 할당의 하드 리밋과 소프트 리밋을 동적으로 변경합니다.
     * WHY: 애플리케이션 실행 중 디바이스 환경에 따라 가용 메모리 한계를 유연하게 재조정하기 위해 사용됩니다.
     * HOW: 전달된 값이 유효한 양수인지, 소프트 리밋이 하드 리밋보다 작거나 같은지 검증한 후 내부 프로퍼티를 갱신합니다.
     */
    setLimits(hardLimitBytes: number, softLimitBytes: number): void;
    /**
     * WHAT: 주어진 크기의 메모리 할당을 예약하고 추적용 토큰을 반환합니다.
     * WHY: 버퍼 생성 전에 쿼터 초과 여부를 먼저 검사하여, 한계 초과 시 안전하게 예외(AMEVAForgeQuotaExceededError)를 발생시키기 위함입니다.
     * HOW: 크기 무결성 검사 후 현재 가용 한계 내인지 확인하고, `allocatedBytes`를 증가시킨 뒤 소프트 리밋을 넘었는지 확인하여 경고합니다. 그 후 새 토큰 객체를 만들어 맵에 등록하고 반환합니다.
     */
    reserveToken(byteLength: number, kind: AllocationKind, ownerGraph?: string | null): AllocationToken;
    /**
     * WHAT: 특정 메모리 토큰을 곧 해제될 것('pending_release')으로 표시합니다.
     * WHY: GPU의 비동기 커맨드 실행이 완료되기 전까지는 버퍼를 파괴할 수 없으므로, 해당 시기를 유예(delay)하면서도 논리적으로는 해제 절차에 들어갔음을 명시하기 위해 존재합니다.
     * HOW: 토큰의 존재와 상태를 검증한 후 상태를 변경하고, `pendingReleaseBytes`에 토큰 크기를 합산합니다.
     */
    markPendingRelease(token: AllocationToken): void;
    /**
     * WHAT: 메모리 토큰이 차지하던 용량을 쿼터 매니저에 완전히 반환하고 토큰을 해제('released') 상태로 바꿉니다.
     * WHY: GPU 리소스가 실제로 해제되었음을 반영하여 가용 메모리(allocatedBytes)를 줄이고 새로운 할당 요청을 수용할 수 있게 하기 위해 필요합니다.
     * HOW: 토큰 상태에 따라 `pendingReleaseBytes`와 `allocatedBytes`를 감소시키고, 토큰을 맵에서 제거합니다.
     */
    releaseToken(token: AllocationToken): void;
    /**
     * WHAT: 현재 메모리 할당량, 대기량, 유효 사용량, 한계치 등의 쿼터 사용 현황을 묶어 반환합니다.
     * WHY: 프로파일러, 디버깅 도구 또는 UI에서 시스템의 메모리 점유 상태를 실시간으로 모니터링하기 위해 제공됩니다.
     * HOW: 클래스 내부에 유지 중인 통계 값(allocated, pending, limits, token size)들을 객체 형태로 복사하여 리턴합니다.
     */
    getUsage(): {
        allocatedBytes: number;
        pendingReleaseBytes: number;
        effectiveBytes: number;
        hardLimitBytes: number;
        softLimitBytes: number;
        activeTokens: number;
    };
    /**
     * WHAT: 메모리 할당 관리의 세대(Generation) 카운터를 1 증가시킵니다.
     * WHY: 실행 그래프나 환경이 크게 전환되는 시점을 마킹하여, 이전 세대에서 생성되었으나 아직 해제되지 않은 누수(Leak) 토큰을 식별하기 위함입니다.
     * HOW: `currentGeneration` 변수에 1을 더합니다.
     */
    incrementGeneration(): void;
    /**
     * WHAT: 현재 할당 관리의 세대 카운터 값을 반환합니다.
     * WHY: 외부 모듈에서 최신 세대 번호를 조회하여 할당 로직이나 상태 리포팅에 활용하기 위해 제공됩니다.
     * HOW: `currentGeneration` 프로퍼티 값을 반환합니다.
     */
    getGeneration(): number;
    /**
     * WHAT: 모든 쿼터 통계치와 관리 중인 토큰을 초기 상태로 되돌립니다.
     * WHY: 테스트 사이의 격리(Isolation)를 보장하거나, 디바이스 초기화 시 이전 상태를 안전하게 파기하기 위해 존재합니다.
     * HOW: 바이트 카운터들을 0으로 설정하고, 토큰 맵을 비웁니다(clear).
     */
    reset(): void;
}
/**
 * WHAT: 전역에서 사용할 수 있는 QuotaManager의 싱글톤 인스턴스입니다.
 * WHY: 애플리케이션 내의 다양한 모듈(버퍼 관리자, 텐서 객체 등)이 하나의 통일된 메모리 한계를 공유하고 갱신하도록 강제하기 위해 생성되었습니다.
 * HOW: QuotaManager를 기본값(1GB/768MB)으로 인스턴스화하여 내보냅니다(export).
 */
export declare const _globalQuotaManager: QuotaManager;
export interface QuotaSnapshot {
    usedBytes: number;
    maxBytes: number;
    activeTokens: number;
}
export declare function getQuotaSnapshot(): QuotaSnapshot;
