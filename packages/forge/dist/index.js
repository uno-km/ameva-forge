(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.ameva = {}));
})(this, (function (exports) { 'use strict';

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    /**
     * WHAT: AMEVA Forge 시스템 전체에서 발생하는 모든 커스텀 에러의 최상위 기본 클래스입니다.
     * WHY: 표준 Error 객체를 확장하여 스택 트레이스와 에러 이름을 올바르게 유지함으로써, 이 라이브러리 내부에서 발생하는 예외 상황을 쉽게 식별하고 포착(catch)할 수 있게 하기 위함입니다.
     * HOW: 자바스크립트의 내장 Error 클래스를 상속(extends)받아 구현됩니다.
     */
    class AMEVAForgeError extends Error {
        /**
         * WHAT: AMEVAForgeError 인스턴스를 생성하는 생성자입니다.
         * WHY: 에러 메시지를 초기화하고, 클래스의 인스턴스 타입 체크(instanceof)가 정상적으로 작동하도록 프로토타입 체인을 교정하기 위해 필요합니다.
         * HOW: 부모 생성자(super)를 호출한 후, this.name을 설정하고 Object.setPrototypeOf를 사용하여 프로토타입을 강제로 맞춰줍니다.
         *
         * @param message 사용자에게 노출될 구체적인 에러 메시지 내용
         */
        constructor(message) {
            super(message);
            this.name = new.target.name;
            Object.setPrototypeOf(this, new.target.prototype);
        }
    }
    /**
     * WHAT: 텐서의 형태(Shape)나 차원(Dimension)이 계산 또는 검증 중 맞지 않을 때 발생하는 에러 클래스입니다.
     * WHY: 연산의 수학적/구조적 조건이 위배되었음을 사용자나 상위 로직에 명확히 알리기 위해 존재합니다.
     * HOW: AMEVAForgeError를 상속받아 정의되어, Shape 관련된 구체적인 예외 상황을 나타내는 타입으로 활용됩니다.
     */
    class AMEVAForgeShapeError extends AMEVAForgeError {
    }
    /**
     * WHAT: 텐서의 데이터 타입(DType)이 연산에서 지원하지 않거나 서로 충돌할 때 발생하는 에러 클래스입니다.
     * WHY: 잘못된 자료형 접근이나 호환되지 않는 텐서 연산을 조기에 차단하여 런타임 크래시를 방지하기 위해 사용됩니다.
     * HOW: AMEVAForgeError를 상속받아 DType 특화 예외를 표현합니다.
     */
    class AMEVAForgeDTypeError extends AMEVAForgeError {
    }
    /**
     * WHAT: GPU 등 하드웨어 디바이스를 초기화하거나 통신하는 과정에서 발생하는 에러 클래스입니다.
     * WHY: 디바이스 손실(Device Lost)이나 잘못된 디바이스 상태 등 하드웨어 의존적인 실패 상황을 명확히 구분하여 처리하기 위해 필요합니다.
     * HOW: AMEVAForgeError를 상속받아 GPU/디바이스 레벨의 문제를 나타냅니다.
     */
    class AMEVAForgeDeviceError extends AMEVAForgeError {
    }
    /**
     * WHAT: 이미 메모리에서 해제된(disposed) 텐서 자원에 접근하려고 시도할 때 발생하는 에러 클래스입니다.
     * WHY: 메모리 누수나 무효한 메모리 접근(Use-After-Free)을 방지하는 안전장치 역할을 하여, 잘못된 리소스 참조를 차단하기 위함입니다.
     * HOW: AMEVAForgeError를 상속받아 생명주기가 끝난 객체에 대한 접근 시 던져집니다.
     */
    class AMEVAForgeDisposedError extends AMEVAForgeError {
    }
    /**
     * WHAT: 시스템이나 WebGPU에서 할당 가능한 메모리 할당량(Quota)이나 버퍼 크기를 초과했을 때 발생하는 에러 클래스입니다.
     * WHY: 제한된 VRAM이나 시스템 리소스 한계에 도달했음을 명확히 알리고, 메모리 할당 실패를 우아하게(gracefully) 처리하기 위해 존재합니다.
     * HOW: AMEVAForgeError를 상속받아 메모리 관련 한계 초과를 나타냅니다.
     */
    class AMEVAForgeQuotaExceededError extends AMEVAForgeError {
    }
    /**
     * WHAT: 실행 중인 브라우저나 환경이 WebGPU API 자체를 지원하지 않을 때 발생하는 에러 클래스입니다.
     * WHY: 호환되지 않는 환경에서 실행을 시도할 때 발생시켜, 폴백(fallback) 메커니즘을 구동하거나 사용자에게 호환성 문제를 신속히 알리기 위해 사용됩니다.
     * HOW: AMEVAForgeError를 상속받아 WebGPU 초기화 실패 시 즉각적으로 던져집니다.
     */
    class AMEVAForgeWebGPUUnavailableError extends AMEVAForgeError {
    }
    /**
     * WHAT: 보안 정책, 권한 부족, 혹은 검증되지 않은 셰이더/WASM 접근 등 보안 관련된 문제가 발생했을 때 던져지는 에러 클래스입니다.
     * WHY: 비정상적인 메모리 접근이나 권한을 벗어난 조작을 막아 시스템의 전반적인 안전성을 보장하기 위한 보호 계층으로 작용합니다.
     * HOW: AMEVAForgeError를 상속받아 보안 정책 위반 시 발동됩니다.
     */
    class AMEVAForgeSecurityError extends AMEVAForgeError {
    }
    /**
     * WHAT: 현재 구현되지 않았거나 지원하지 않는 연산(Operation)을 실행하려고 할 때 발생하는 에러 클래스입니다.
     * WHY: 사용자가 유효하지 않은 그래프 노드나 현재 라이브러리에서 지원 범위를 벗어난 커널을 호출하는 것을 사전에 막아 오작동을 예방합니다.
     * HOW: AMEVAForgeError를 상속받아 구현되지 않은 기능 호출 시 발생합니다.
     */
    class AMEVAForgeUnsupportedOpError extends AMEVAForgeError {
    }
    /**
     * WHAT: GPU validation error scope에서 감지된 오류 클래스입니다.
     * WHY: WebGPU pushErrorScope('validation') 결과를 typed exception으로 전달하기 위해 존재합니다.
     */
    class AMEVAForgeValidationError extends AMEVAForgeError {
    }
    /**
     * WHAT: GPU out-of-memory error scope에서 감지된 오류 클래스입니다.
     * WHY: WebGPU pushErrorScope('out-of-memory') 결과를 typed exception으로 전달하기 위해 존재합니다.
     */
    class AMEVAForgeOutOfMemoryError extends AMEVAForgeError {
    }
    /**
     * WHAT: GPU internal error scope에서 감지된 오류 클래스입니다.
     * WHY: WebGPU pushErrorScope('internal') 결과를 typed exception으로 전달하기 위해 존재합니다.
     */
    class AMEVAForgeInternalGPUError extends AMEVAForgeError {
    }
    /**
     * WHAT: GPU 디바이스가 유실(device lost)되었을 때 발생하는 오류 클래스입니다.
     * WHY: 디바이스 유실 상황을 명확히 구분하여 재초기화 흐름을 유도하기 위해 존재합니다.
     */
    class AMEVAForgeDeviceLostError extends AMEVAForgeError {
    }
    /**
     * WHAT: 이전 generation의 stale handle에 접근할 때 발생하는 오류 클래스입니다.
     * WHY: Device lost 후 재초기화된 환경에서 이전 텐서 접근을 차단하기 위해 존재합니다.
     */
    class AMEVAForgeStaleHandleError extends AMEVAForgeError {
    }

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
     * WHAT: 할당된 GPU 메모리 블록을 추적하는 메타데이터 객체(토큰) 클래스입니다.
     * WHY: 실제 GPU 버퍼 리소스와 매핑되어 해당 할당의 크기, 종류, 소유자 등을 식별하고 QuotaManager를 통한 반환을 제어하기 위해 존재합니다.
     * HOW: 생성자에서 유니크 ID, 크기(size), 종류, 그래프 소유자(ownerGraph) 및 생성 세대(generation)를 주입받아 초기화합니다.
     */
    class AllocationToken {
        id;
        size;
        kind;
        state;
        ownerGraph;
        generation;
        /**
         * WHAT: AllocationToken 인스턴스를 초기화하는 생성자입니다.
         * WHY: 새로운 메모리 할당이 예약될 때 필요한 상태 및 식별 데이터를 객체에 부여하기 위해 호출됩니다.
         * HOW: 전달받은 파라미터로 멤버 변수를 초기화하며, 초기 상태는 'active'로 설정합니다.
         */
        constructor(id, size, kind, ownerGraph, generation) {
            this.id = id;
            this.size = size;
            this.kind = kind;
            this.state = 'active';
            this.ownerGraph = ownerGraph;
            this.generation = generation;
        }
    }
    /**
     * WHAT: 시스템 전체의 GPU VRAM 할당량을 중앙 집중적으로 관리하는 클래스입니다.
     * WHY: WebGPU 애플리케이션에서 발생할 수 있는 Out-Of-Memory(OOM) 오류를 미연에 방지하고, 메모리 누수를 추적하며 동적으로 한계(Limits)를 설정하기 위해 필요합니다.
     * HOW: 소프트 제한(soft limit)과 하드 제한(hard limit)을 기반으로 메모리 할당 요청을 검사하고 허용하거나 거부하며, 할당된 토큰을 Map을 통해 상태별로 관리합니다.
     */
    class QuotaManager {
        /**
         * WHAT: 현재 활성 상태로 할당된 총 메모리 크기(바이트)입니다.
         * WHY: 사용 중인 리소스를 합산하여 쿼터를 초과하지 않는지 감시하기 위해 유지합니다.
         * HOW: reserveToken 시 증가하고, releaseToken 시 감소합니다.
         */
        allocatedBytes = 0;
        /**
         * WHAT: 해제가 예약되었으나 아직 완전히 반환되지 않은 메모리 크기(바이트)입니다.
         * WHY: 비동기 작업 중 잠시 유지되는 메모리를 계산하여 여유 한계를 보다 정확하게 산정하기 위함입니다.
         * HOW: markPendingRelease 호출 시 증가하고, 완전히 해제될 때 감소합니다.
         */
        pendingReleaseBytes = 0;
        /**
         * WHAT: 메모리 할당이 절대로 초과할 수 없는 최대 허용치(바이트)입니다.
         * WHY: 이 값을 초과하는 할당 요청을 즉시 차단하여 치명적인 시스템 충돌(OOM)을 막기 위해 설정됩니다.
         * HOW: 생성자에서 주입되거나 setLimits를 통해 설정됩니다.
         */
        hardLimitBytes;
        /**
         * WHAT: 경고를 발생시키는 메모리 사용량의 임계점(바이트)입니다.
         * WHY: 하드 리밋에 도달하기 전 시스템에 과부하가 올 수 있음을 경고(warn)하기 위해 사용됩니다.
         * HOW: 실제 사용량(allocated - pending)이 이 값을 초과할 때 콘솔에 경고 로그를 출력합니다.
         */
        softLimitBytes;
        /**
         * WHAT: 발급된 모든 AllocationToken을 고유 식별자(ID)로 관리하는 맵(Map)입니다.
         * WHY: 토큰의 무결성 검증, 이중 해제(Double Free) 방지 및 전체 할당 현황 조회를 위해 존재합니다.
         * HOW: 토큰 발급 시 추가하고 해제 시 삭제합니다.
         */
        tokens = new Map();
        /**
         * WHAT: 새로 생성되는 메모리 토큰에 부여할 고유 식별자 카운터입니다.
         * WHY: 각 할당 토큰을 구별하고 맵에서 충돌 없이 관리하기 위해 필요합니다.
         * HOW: 새로운 토큰이 생성될 때마다 1씩 증가합니다.
         */
        nextId = 1;
        /**
         * WHAT: 현재 할당 주기를 나타내는 세대(Generation) 카운터입니다.
         * WHY: 그래프 재컴파일 등 대규모 변경이 일어날 때 이전 세대의 토큰들을 구분하고 메모리 누수를 진단하기 위해 도입되었습니다.
         * HOW: incrementGeneration() 호출 시 증가하며 토큰 생성 시 부여됩니다.
         */
        currentGeneration = 1;
        /**
         * WHAT: QuotaManager 클래스의 인스턴스를 초기화하는 생성자입니다.
         * WHY: 객체 생성 시 초기 하드 리밋과 소프트 리밋 용량을 설정하기 위해 호출됩니다.
         * HOW: 전달된 바이트 값을 각각의 클래스 프로퍼티에 할당합니다.
         */
        constructor(hardLimitBytes = 1 * 1024 * 1024 * 1024, softLimitBytes = 768 * 1024 * 1024) {
            this.hardLimitBytes = hardLimitBytes;
            this.softLimitBytes = softLimitBytes;
        }
        /**
         * WHAT: 메모리 할당의 하드 리밋과 소프트 리밋을 동적으로 변경합니다.
         * WHY: 애플리케이션 실행 중 디바이스 환경에 따라 가용 메모리 한계를 유연하게 재조정하기 위해 사용됩니다.
         * HOW: 전달된 값이 유효한 양수인지, 소프트 리밋이 하드 리밋보다 작거나 같은지 검증한 후 내부 프로퍼티를 갱신합니다.
         */
        setLimits(hardLimitBytes, softLimitBytes) {
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
        /**
         * WHAT: 주어진 크기의 메모리 할당을 예약하고 추적용 토큰을 반환합니다.
         * WHY: 버퍼 생성 전에 쿼터 초과 여부를 먼저 검사하여, 한계 초과 시 안전하게 예외(AMEVAForgeQuotaExceededError)를 발생시키기 위함입니다.
         * HOW: 크기 무결성 검사 후 현재 가용 한계 내인지 확인하고, `allocatedBytes`를 증가시킨 뒤 소프트 리밋을 넘었는지 확인하여 경고합니다. 그 후 새 토큰 객체를 만들어 맵에 등록하고 반환합니다.
         */
        reserveToken(byteLength, kind, ownerGraph = null) {
            if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
                throw new AMEVAForgeQuotaExceededError(`Invalid allocation size: ${byteLength}`);
            }
            if (byteLength > this.hardLimitBytes - this.allocatedBytes) {
                throw new AMEVAForgeQuotaExceededError(`Quota Exceeded: Cannot allocate ${byteLength} bytes. ` +
                    `Current: ${this.allocatedBytes} (${this.pendingReleaseBytes} pending release), ` +
                    `Limit: ${this.hardLimitBytes}`);
            }
            this.allocatedBytes += byteLength;
            if (this.allocatedBytes - this.pendingReleaseBytes > this.softLimitBytes) {
                console.warn(`[AMEVA] VRAM soft quota exceeded: ` +
                    `${((this.allocatedBytes - this.pendingReleaseBytes) / 1e9).toFixed(2)}GB / ` +
                    `${(this.softLimitBytes / 1e9).toFixed(2)}GB`);
            }
            const id = `alloc_${this.nextId++}`;
            const token = new AllocationToken(id, byteLength, kind, ownerGraph, this.currentGeneration);
            this.tokens.set(id, token);
            return token;
        }
        /**
         * WHAT: 특정 메모리 토큰을 곧 해제될 것('pending_release')으로 표시합니다.
         * WHY: GPU의 비동기 커맨드 실행이 완료되기 전까지는 버퍼를 파괴할 수 없으므로, 해당 시기를 유예(delay)하면서도 논리적으로는 해제 절차에 들어갔음을 명시하기 위해 존재합니다.
         * HOW: 토큰의 존재와 상태를 검증한 후 상태를 변경하고, `pendingReleaseBytes`에 토큰 크기를 합산합니다.
         */
        markPendingRelease(token) {
            if (!token || token.state !== 'active')
                return;
            // Verify token exists and belongs to us
            if (!this.tokens.has(token.id))
                return;
            token.state = 'pending_release';
            const newPending = this.pendingReleaseBytes + token.size;
            this.pendingReleaseBytes = Math.min(newPending, this.allocatedBytes);
        }
        /**
         * WHAT: 메모리 토큰이 차지하던 용량을 쿼터 매니저에 완전히 반환하고 토큰을 해제('released') 상태로 바꿉니다.
         * WHY: GPU 리소스가 실제로 해제되었음을 반영하여 가용 메모리(allocatedBytes)를 줄이고 새로운 할당 요청을 수용할 수 있게 하기 위해 필요합니다.
         * HOW: 토큰 상태에 따라 `pendingReleaseBytes`와 `allocatedBytes`를 감소시키고, 토큰을 맵에서 제거합니다.
         */
        releaseToken(token) {
            if (!token || token.state === 'released')
                return;
            if (!this.tokens.has(token.id))
                return;
            if (token.state === 'pending_release') {
                this.pendingReleaseBytes = Math.max(0, this.pendingReleaseBytes - token.size);
            }
            this.allocatedBytes = Math.max(0, this.allocatedBytes - token.size);
            token.state = 'released';
            this.tokens.delete(token.id);
        }
        /**
         * WHAT: 현재 메모리 할당량, 대기량, 유효 사용량, 한계치 등의 쿼터 사용 현황을 묶어 반환합니다.
         * WHY: 프로파일러, 디버깅 도구 또는 UI에서 시스템의 메모리 점유 상태를 실시간으로 모니터링하기 위해 제공됩니다.
         * HOW: 클래스 내부에 유지 중인 통계 값(allocated, pending, limits, token size)들을 객체 형태로 복사하여 리턴합니다.
         */
        getUsage() {
            return {
                allocatedBytes: this.allocatedBytes,
                pendingReleaseBytes: this.pendingReleaseBytes,
                effectiveBytes: this.allocatedBytes - this.pendingReleaseBytes,
                hardLimitBytes: this.hardLimitBytes,
                softLimitBytes: this.softLimitBytes,
                activeTokens: this.tokens.size
            };
        }
        /**
         * WHAT: 메모리 할당 관리의 세대(Generation) 카운터를 1 증가시킵니다.
         * WHY: 실행 그래프나 환경이 크게 전환되는 시점을 마킹하여, 이전 세대에서 생성되었으나 아직 해제되지 않은 누수(Leak) 토큰을 식별하기 위함입니다.
         * HOW: `currentGeneration` 변수에 1을 더합니다.
         */
        incrementGeneration() {
            this.currentGeneration++;
        }
        /**
         * WHAT: 현재 할당 관리의 세대 카운터 값을 반환합니다.
         * WHY: 외부 모듈에서 최신 세대 번호를 조회하여 할당 로직이나 상태 리포팅에 활용하기 위해 제공됩니다.
         * HOW: `currentGeneration` 프로퍼티 값을 반환합니다.
         */
        getGeneration() {
            return this.currentGeneration;
        }
        /**
         * WHAT: 모든 쿼터 통계치와 관리 중인 토큰을 초기 상태로 되돌립니다.
         * WHY: 테스트 사이의 격리(Isolation)를 보장하거나, 디바이스 초기화 시 이전 상태를 안전하게 파기하기 위해 존재합니다.
         * HOW: 바이트 카운터들을 0으로 설정하고, 토큰 맵을 비웁니다(clear).
         */
        reset() {
            this.allocatedBytes = 0;
            this.pendingReleaseBytes = 0;
            this.tokens.clear();
        }
    }
    /**
     * WHAT: 전역에서 사용할 수 있는 QuotaManager의 싱글톤 인스턴스입니다.
     * WHY: 애플리케이션 내의 다양한 모듈(버퍼 관리자, 텐서 객체 등)이 하나의 통일된 메모리 한계를 공유하고 갱신하도록 강제하기 위해 생성되었습니다.
     * HOW: QuotaManager를 기본값(1GB/768MB)으로 인스턴스화하여 내보냅니다(export).
     */
    const _globalQuotaManager = new QuotaManager();
    function getQuotaSnapshot() {
        const usage = _globalQuotaManager.getUsage();
        return Object.freeze({
            usedBytes: usage.allocatedBytes,
            maxBytes: usage.hardLimitBytes,
            activeTokens: usage.activeTokens,
        });
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * device.ts — WebGPU 싱글톤 디바이스 래퍼
     *
     * H-04 Fix: getAdapter() export 추가 → gpuCore.ts에서 adapter.limits 조회 가능
     * L-03 Fix: device lost 시 onDeviceLostCallback을 통해 pipelineCache도 무효화
     */
    /**
     * WHAT: 개발 환경이나 디버그 모드에서만 시스템 메시지를 출력하는 안전한 로깅 함수입니다.
     * WHY: 불필요한 콘솔 출력을 프로덕션 환경에서 방지하고, 에러 없이 안전하게 로그를 남기기 위해 사용됩니다.
     * HOW: 현재 실행 환경이 개발 모드(NODE_ENV, AMEVA_DEBUG, __DEV__, Vite env 등)인지 확인하고 조건을 만족할 때만 `globalThis.log`를 통해 메시지를 출력합니다. 예외가 발생해도 시스템이 멈추지 않도록 try-catch로 감쌉니다.
     */
    function _safeLog$1(msg) {
        try {
            // VUL-015 Fix: Only log in development or explicit debug modes
            const isDev = (typeof process !== 'undefined' && process.env && process.env.NODE_ENV !== 'production') ||
                (typeof globalThis.AMEVA_DEBUG !== 'undefined' && globalThis.AMEVA_DEBUG) ||
                (typeof globalThis.__DEV__ !== 'undefined' && globalThis.__DEV__);
            // Vite/ESBuild injects import.meta.env, wrap in try-catch to avoid syntax errors in older environments
            let isViteDev = false;
            try {
                const getImportMeta = new Function('return import.meta');
                const meta = getImportMeta();
                isViteDev = meta && meta.env && meta.env.MODE !== 'production';
            }
            catch (e) { }
            if (!isDev && !isViteDev)
                return;
            if (typeof globalThis.log === 'function') {
                globalThis.log(msg, 'system');
            }
        }
        catch (e) { }
    }
    /**
     * WHAT: 초기화된 논리적 WebGPU 디바이스(GPUDevice) 인스턴스를 저장하는 내부 변수입니다.
     * WHY: 모듈 내에서 싱글톤(singleton) 패턴을 유지하여 여러 번 초기화되지 않도록 상태를 관리합니다.
     * HOW: initWebGPU 함수 내에서 생성된 디바이스가 할당되며, 디바이스 손실(device lost) 시 다시 null로 초기화됩니다.
     */
    let device = null;
    /**
     * WHAT: WebGPU 기능 및 하드웨어 한계를 나타내는 물리적 GPU 어댑터(GPUAdapter) 인스턴스를 저장하는 내부 변수입니다.
     * WHY: 디바이스 생성 전 스펙(limits)을 검사하거나 생성 후 하드웨어 정보를 참조하기 위해 캐싱해 둡니다.
     * HOW: initWebGPU 실행 시 requestAdapter()로 요청받아 할당되며, 디바이스 손실 시 함께 정리됩니다.
     */
    let adapter = null;
    /**
     * WHAT: 디바이스 손실(device lost) 발생 시 실행될 콜백 함수를 저장하는 변수입니다.
     * WHY: GPU 오류나 컨텍스트 초기화 상황이 발생했을 때 상위 애플리케이션으로 이벤트를 위임하기 위해 필요합니다.
     * HOW: setDeviceLostCallback 함수를 통해 설정되며, device.lost Promise가 해결(resolve)될 때 내부적으로 호출됩니다.
     */
    let onDeviceLostCallback = null;
    /**
     * WHAT: 시스템 환경에서 WebGPU 디바이스 및 어댑터를 비동기적으로 초기화합니다.
     * WHY: WebGPU API를 사용하기 위해 필수적인 하드웨어 어댑터(adapter)와 논리적 디바이스(device) 인스턴스를 확보하고 전역에서 접근할 수 있도록 캐싱하기 위해 존재합니다.
     * HOW:
     *   1. navigator.gpu 객체가 존재하는지 확인하고, requestAdapter()로 물리적 GPU 어댑터를 요청합니다.
     *   2. 어댑터가 지원하는 최대 버퍼 크기 등의 한계를 파악하여 requestDevice()로 디바이스를 생성합니다.
     *   3. 디바이스 손실(device.lost) 이벤트를 수신하여 리소스를 정리하고 등록된 콜백을 실행하도록 설정합니다.
     */
    async function initWebGPU(options) {
        _safeLog$1(`[device.ts] initWebGPU started. current device=${device ? 'SET' : 'NULL'}`);
        if (device)
            return;
        if (typeof navigator === "undefined" || !navigator.gpu) {
            throw new AMEVAForgeWebGPUUnavailableError("WebGPU is not available in this environment. " +
                "Ensure you are running in a supported browser with WebGPU enabled.");
        }
        adapter = await navigator.gpu.requestAdapter(options);
        if (!adapter) {
            throw new AMEVAForgeWebGPUUnavailableError("Failed to request a WebGPU adapter. " +
                "Your GPU may not support WebGPU, or the browser has disabled it.");
        }
        const requiredLimits = {};
        if (adapter.limits) {
            requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
            requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
        }
        device = await adapter.requestDevice({ requiredLimits });
        if (device.limits && device.limits.maxStorageBufferBindingSize) {
            const maxBinding = device.limits.maxStorageBufferBindingSize;
            const adaptedHard = Math.max(1024 * 1024 * 1024, maxBinding * 2);
            const adaptedSoft = Math.max(768 * 1024 * 1024, maxBinding);
            try {
                _globalQuotaManager.setLimits(adaptedHard, adaptedSoft);
            }
            catch { }
        }
        _safeLog$1(`[device.ts] initWebGPU finished. device successfully created.`);
        device.lost.then((info) => {
            const msg = `[AMEVA] WebGPU Device Lost: ${info.message} (reason: ${info.reason})`;
            console.error(msg);
            _safeLog$1(msg);
            device = null;
            // (globalThis as any).__AMEVA_DEVICE__ = null;
            adapter = null;
            if (onDeviceLostCallback) {
                onDeviceLostCallback();
            }
        });
    }
    /**
     * WHAT: 전역에 캐시된 WebGPU 디바이스 인스턴스를 반환합니다.
     * WHY: 애플리케이션의 여러 모듈에서 동일한 단일 디바이스 인스턴스에 접근하여 버퍼 및 텍스처를 생성할 수 있도록 제공하기 위함입니다.
     * HOW: 내부 `device` 변수가 초기화되어 있는지 확인하고, 없을 경우 예외(AMEVAForgeDeviceError)를 발생시키며, 존재할 경우 그대로 반환합니다.
     */
    function getDevice() {
        if (!device) {
            const globalExists = typeof globalThis.amevaForge !== "undefined";
            throw new AMEVAForgeDeviceError(`WebGPU device is not initialized. (globalThis.amevaForge exists: ${globalExists}). Call await init() first.`);
        }
        return device;
    }
    /**
     * WHAT: 전역에 캐시된 WebGPU 어댑터(Adapter) 인스턴스를 반환합니다.
     * WHY: GPU의 하드웨어 스펙(limits, features 등)을 조회하거나 디바이스 기능 제약 조건을 파악하기 위해 외부 모듈에서 어댑터에 접근할 수 있게 합니다.
     * HOW: 내부 `adapter` 변수를 그대로 반환합니다. 아직 초기화되지 않았다면 null이 반환될 수 있습니다.
     */
    function getAdapter() {
        return adapter;
    }
    /**
     * WHAT: 초기화된 WebGPU 디바이스와 연결된 커맨드 큐(GPUQueue)를 반환합니다.
     * WHY: 데이터를 버퍼로 전송(writeBuffer)하거나 렌더링/컴퓨트 커맨드(submit)를 실행할 수 있도록 접근 지점을 제공합니다.
     * HOW: `getDevice()` 함수를 호출해 디바이스를 얻은 후 `device.queue` 속성을 반환합니다.
     */
    function getQueue() {
        return getDevice().queue;
    }
    /**
     * WHAT: WebGPU 디바이스가 현재 성공적으로 초기화되어 사용 가능한지 여부를 반환합니다.
     * WHY: 기능 호환성 검사나 런타임 조건부 로직 실행 전, WebGPU 사용 가능 여부를 안전하게 확인하기 위해 제공됩니다.
     * HOW: 내부에 저장된 `device` 변수가 null이 아닌지 불리언(Boolean) 값으로 평가하여 반환합니다.
     */
    function isAvailable() {
        return device !== null;
    }
    function _resetDeviceForTesting() {
        device = null;
        adapter = null;
        if (onDeviceLostCallback) {
            onDeviceLostCallback();
        }
    }
    /**
     * WHAT: GPU 디바이스 연결이 끊어졌을 때(device lost) 호출될 콜백 함수를 등록합니다.
     * WHY: 예기치 못한 GPU 충돌이나 컨텍스트 상실 시 상위 계층(예: 파이프라인 캐시 무효화, 재초기화 로직)에 이를 알리기 위해 존재합니다.
     * HOW: 전달받은 함수(callback)를 모듈 레벨 변수인 `onDeviceLostCallback`에 할당하여 이후 디바이스 손실 이벤트 발생 시 실행될 수 있도록 합니다.
     */
    function setDeviceLostCallback(callback) {
        onDeviceLostCallback = callback;
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * WHAT: WASM(WebAssembly) 메모리 영역에 접근할 때 오프셋과 길이의 유효성을 검증하는 모듈입니다.
     * WHY: 잘못된 메모리 주소나 범위를 참조하여 발생하는 버퍼 오버플로우, 세그멘테이션 폴트 및 잠재적 보안 취약점을 차단하기 위해 필요합니다.
     */
    /**
     * WHAT: 주어진 오프셋과 데이터 길이가 WASM 선형 메모리 힙(heap)의 유효한 범위 내에 있는지 안전하게 검사합니다.
     * WHY: CPU-GPU 간 데이터 전송이나 공유 메모리 접근 시 악의적이거나 잘못된 크기 요청으로 인한 메모리 침범을 방어하기 위해 호출됩니다.
     * HOW: `Number.isSafeInteger`와 비음수(non-negative) 조건을 통해 입력 인자의 데이터 타입을 엄격히 검증한 후, `offset + byteLength`가 총 WASM 메모리 크기를 초과하지 않는지 계산하여 확인합니다. 위반 시 보안 예외를 던집니다.
     */
    function assertWasmRange(offset, byteLength, wasmByteLength) {
        if (!Number.isSafeInteger(offset) || offset < 0) {
            throw new AMEVAForgeSecurityError("Invalid offset: must be a non-negative safe integer.");
        }
        if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
            throw new AMEVAForgeSecurityError("Invalid byteLength: must be a non-negative safe integer.");
        }
        if (!Number.isSafeInteger(wasmByteLength) || wasmByteLength < 0) {
            throw new AMEVAForgeSecurityError("Invalid wasmByteLength: must be a non-negative safe integer.");
        }
        if (offset > wasmByteLength || byteLength > wasmByteLength - offset) {
            throw new AMEVAForgeSecurityError("WASM memory range out of bounds");
        }
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * shaderGuard.ts — WGSL 셰이더 보안 가드
     *
     * H-07 Fix: 화이트리스트에 모든 구현된 op 추가.
     * NH-07 Fix: 이 파일의 assertAllowedKernelName()을 graphExecutor.ts와 gpuCore.ts에서
     *   실제로 import하여 사용한다 (이전에는 데드 코드였음).
     */
    /**
     * WHAT: 셰이더 내에서 사용될 식별자(함수명, 변수명)가 안전한 문자열인지 검사합니다.
     * WHY: 영숫자와 밑줄(_) 이외의 문자가 주입되어 비정상적인 코드 실행이나 컴파일 에러를 유도하는 셰이더 인젝션 공격을 예방하기 위해 존재합니다.
     * HOW: 정규 표현식(/^[a-zA-Z_][a-zA-Z0-9_]*$/)을 사용하여 문자열 패턴을 검증하고, 실패 시 AMEVAForgeSecurityError를 던집니다.
     */
    function assertSafeShaderIdentifier(identifier) {
        if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
            throw new AMEVAForgeSecurityError(`Invalid shader identifier: "${identifier}". Only alphanumeric and underscore allowed.`);
        }
    }
    /**
     * WHAT: 셰이더에 주입되는 상수(숫자) 값이 유한한(finite) 숫자인지 확인합니다.
     * WHY: Infinity나 NaN과 같은 유효하지 않은 값이 셰이더 소스에 포함되어 GPU 연산 오류를 유발하는 것을 막기 위함입니다.
     * HOW: `Number.isFinite(value)`로 검사하고, 유한하지 않은 경우 예외를 발생시킵니다.
     */
    function assertAllowedShaderConstant(value) {
        if (!Number.isFinite(value)) {
            throw new AMEVAForgeSecurityError(`Invalid shader constant: ${value}. Must be a finite number.`);
        }
    }
    /**
     * WHAT: 전달된 셰이더 소스 문자열에 동적 템플릿 리터럴 구문("${" 또는 "`")이 포함되어 있는지 검사합니다.
     * WHY: 신뢰할 수 없는 데이터가 셰이더 코드로 동적으로 삽입되는 인젝션 공격(Template Literal Injection)을 철저히 차단하기 위해 필요합니다.
     * HOW: 문자열의 `includes` 메서드를 통해 해당 패턴의 존재 여부를 검사하고, 발견될 경우 에러를 던집니다.
     */
    function assertStaticShaderSourceOnly(source) {
        if (source.includes("${") || source.includes("`")) {
            throw new AMEVAForgeSecurityError("Dynamic shader source interpolation is forbidden. Use uniform buffers for runtime values.");
        }
    }
    /**
     * WHAT: 보안 상 실행이 허용된 커널(연산) 이름들을 저장하는 화이트리스트(Set)입니다.
     * WHY: 허가되지 않은 임의의 커널 이름이 실행 경로로 주입되어 예상치 못한 셰이더 모듈이 생성되거나 호출되는 보안 취약점을 방어하기 위해 존재합니다.
     * HOW: Set 자료구조로 초기화하여 허용 목록을 빠르게 조회(has)할 수 있도록 합니다.
     *
     * H-07/NH-07 Fix: 모든 구현된 커널 이름을 화이트리스트에 포함.
     * graphExecutor.ts의 ALLOWED_OPS와 반드시 동기화 유지.
     * 이 함수는 gpuCore.ts와 graphExecutor.ts에서 실제로 호출된다.
     */
    let ALLOWED_KERNEL_NAMES = new Set([
        "matmul",
        "relu",
        "relu_backward",
        "add",
        "mul",
        "transpose",
        // v2.0: 학습 기능에 필요한 커널 추가 (VUL-001 Fix)
        "sub",
        "neg",
        "div",
        "exp",
        "log",
        "sigmoid",
        "tanh",
        "sigmoid_backward",
        "tanh_backward",
        "fill",
        "sum",
        "max",
        "sum_axis",
        "axpy",
        "pad",
        "gather",
        "scatter",
        "dropout",
        "maxpool2d",
        "avgpool2d",
        "im2col",
        "col2im",
        "permute",
        "matmul_bias_relu",
    ]);
    /**
     * WHAT: 화이트리스트에 허용된 커널 이름들을 새롭게 등록(덮어쓰기)합니다.
     * WHY: 애플리케이션 초기화 단계 또는 플러그인 로드 시 동적으로 안전한 커널 목록을 확장하고 갱신할 수 있도록 유연성을 제공하기 위함입니다.
     * HOW: 제공된 Iterable 인터페이스(예: 배열)를 받아 새로운 Set 객체를 생성하고 `ALLOWED_KERNEL_NAMES` 변수를 갱신합니다.
     */
    function registerKernelNames(names) {
        ALLOWED_KERNEL_NAMES = new Set(names);
    }
    /**
     * WHAT: 요청된 커널 이름이 허용된 화이트리스트(ALLOWED_KERNEL_NAMES)에 존재하는지 검사합니다.
     * WHY: 그래프 실행기(graphExecutor)나 GPU 코어 모듈이 연산을 수행하기 직전, 허가되지 않은 커널 호출을 차단하기 위해 사용됩니다.
     * HOW: `Set.has(name)` 메서드를 사용하여 포함 여부를 확인하고 없으면 보안 예외(SecurityError)를 발생시킵니다.
     */
    function assertAllowedKernelName(name) {
        if (!ALLOWED_KERNEL_NAMES.has(name)) {
            throw new AMEVAForgeSecurityError(`Unknown kernel name: "${name}". Allowed: ${[...ALLOWED_KERNEL_NAMES].join(", ")}`);
        }
    }
    /**
     * WHAT: 현재 설정된 커널 이름 화이트리스트(Set)의 읽기 전용 참조를 반환합니다.
     * WHY: 외부 모듈에서 화이트리스트의 구성을 확인할 수 있게 하면서도 직접적인 데이터 변조는 방지하기 위해 존재합니다.
     * HOW: 모듈 내부의 `ALLOWED_KERNEL_NAMES` 변수를 ReadonlySet 타입으로 캐스팅하여 그대로 반환합니다.
     */
    function getAllowedKernelNames() {
        return ALLOWED_KERNEL_NAMES;
    }

    /**
     * Created: 2026-08-12T12:14:52+09:00
     * Modified:
     *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    /**
     * WHAT: dtype별 바이트 크기를 매핑하는 상수 딕셔너리입니다.
     * WHY: 텐서의 전체 바이트 크기를 계산할 때, 데이터 타입마다 차지하는 바이트 수가 다르기 때문에 이를 정확히 계산하기 위해 존재합니다.
     * HOW: Record 유틸리티 타입을 사용하여 DType 문자열을 키로, 바이트 수(number)를 값으로 갖는 객체를 정의합니다.
     */
    const BYTES_PER_ELEMENT = {
        "float32": 4,
        // float16: 2 — 셰이더 구현 완료 후 추가 예정
        // int32: 4 — 셰이더 구현 완료 후 추가 예정
    };
    /**
     * WHAT: 텐서가 가질 수 있는 최대 원소 수를 정의하는 상수입니다.
     * WHY: 메모리 초과(OOM) 오류를 방지하고 시스템의 안정성을 유지하기 위해 하드 리미트를 설정합니다.
     * HOW: 256MB 크기의 float32 버퍼에 맞추어 256 * 1024 * 1024로 값을 할당합니다.
     */
    const MAX_ELEMENTS$1 = 256 * 1024 * 1024;
    /**
     * WHAT: 텐서 shape의 최대 랭크(차원 수)를 정의하는 상수입니다.
     * WHY: WebGPU에서 처리할 수 있는 차원의 한계를 설정하고, 과도하게 복잡한 다차원 텐서의 생성을 방지합니다.
     * HOW: 스칼라(rank 0)부터 시작하여 최대 8차원까지 허용하도록 숫자 8을 할당합니다.
     */
    const MAX_RANK = 8; // NM-06: 스칼라(rank 0) 포함하여 0~8까지 허용
    /**
     * WHAT: 텐서 shape의 유효성을 검증하고 총 원소 수를 반환하는 함수입니다.
     * WHY: 잘못된 텐서 형태나 예상치 못한 크기의 메모리 할당을 사전에 차단하여 안전한 연산을 보장하기 위함입니다.
     * HOW: 입력된 shape가 배열인지, 랭크 제한을 넘지 않는지 확인한 후, 각 차원의 값을 곱해 총 원소 수를 구합니다. 예상 바이트 크기가 주어진 경우 이를 함께 검증합니다.
     *
     * M-01 Fix: dtype별 바이트 크기를 BYTES_PER_ELEMENT 맵으로 정확히 계산.
     * NM-06 Fix: rank 0 스칼라 텐서 허용 (PyTorch/JAX/TF 표준).
     *   rank 0 = shape=[], elements=1, byteLength=4 (단일 float32 스칼라)
     */
    function validateShape(shape, dtype, expectedByteLength) {
        if (!Array.isArray(shape)) {
            throw new AMEVAForgeShapeError("Shape must be an array.");
        }
        // NM-06 Fix: rank 0 (shape=[]) 허용 — 스칼라 텐서
        if (shape.length > MAX_RANK) {
            throw new AMEVAForgeShapeError(`Shape rank must be between 0 and ${MAX_RANK}, got ${shape.length}.`);
        }
        /**
         * WHAT: 텐서의 총 원소 수를 누적하여 저장하는 변수입니다.
         * WHY: shape 배열의 각 차원을 곱하여 텐서 데이터가 차지할 실제 원소의 총 개수를 알아내기 위해 필요합니다.
         * HOW: 스칼라(rank 0)의 경우를 처리하기 위해 1로 초기화됩니다.
         */
        let elements = 1;
        /**
         * WHAT: shape 배열의 각 차원 값을 순회하며 원소 수를 계산하고 유효성을 검사하는 루프입니다.
         * WHY: 모든 차원의 값이 양의 정수인지 확인하고, 안전한 정수 범위를 벗어나는 오버플로우를 감지하기 위해 존재합니다.
         * HOW: 인덱스 i를 0부터 shape.length - 1까지 증가시키며 dim 값을 추출해 검증하고 elements에 누적 곱셈을 수행합니다.
         */
        for (let i = 0; i < shape.length; i++) {
            /**
             * WHAT: 현재 검사 중인 텐서의 특정 차원(dimension)의 크기를 나타내는 변수입니다.
             * WHY: 이 값이 유효한 양의 정수인지 검사하기 위해 루프 내에서 임시로 저장합니다.
             * HOW: shape 배열에서 i번째 인덱스의 값을 참조하여 가져옵니다.
             */
            const dim = shape[i];
            if (!Number.isSafeInteger(dim) || dim <= 0) {
                throw new AMEVAForgeShapeError(`shape[${i}] must be positive, got ${dim}`);
            }
            if (dim > Number.MAX_SAFE_INTEGER / elements) {
                throw new AMEVAForgeShapeError("Shape product overflows safe integer limit.");
            }
            elements *= dim;
        }
        if (elements > MAX_ELEMENTS$1) {
            throw new AMEVAForgeShapeError(`Tensor size exceeds max elements limit: ${elements} > ${MAX_ELEMENTS$1}`);
        }
        if (expectedByteLength !== undefined) {
            /**
             * WHAT: 입력된 dtype이 차지하는 단일 원소의 바이트 크기를 저장하는 변수입니다.
             * WHY: 전체 텐서의 예상 바이트 크기를 계산하기 위해 요소당 크기를 알아야 합니다.
             * HOW: BYTES_PER_ELEMENT 상수 맵에서 dtype을 키로 사용하여 값을 조회합니다.
             */
            const bytesPerElement = BYTES_PER_ELEMENT[dtype];
            if (bytesPerElement === undefined) {
                throw new AMEVAForgeDTypeError(`Unsupported dtype for byte size calculation: "${dtype}". ` +
                    `Supported: ${Object.keys(BYTES_PER_ELEMENT).join(', ')}`);
            }
            /**
             * WHAT: shape와 dtype을 바탕으로 계산된 텐서의 실제 필요 바이트 크기를 담는 변수입니다.
             * WHY: 사용자가 제시한 expectedByteLength와 비교하여 데이터 정합성을 검증하기 위해 계산합니다.
             * HOW: 누적된 총 원소 수(elements)에 원소당 바이트 크기(bytesPerElement)를 곱하여 구합니다.
             */
            const calculatedBytes = elements * bytesPerElement;
            if (calculatedBytes !== expectedByteLength) {
                throw new AMEVAForgeShapeError(`Shape/data size mismatch: shape ${JSON.stringify(shape)} (${dtype}) ` +
                    `implies ${calculatedBytes} bytes, but data is ${expectedByteLength} bytes.`);
            }
        }
        return elements;
    }

    /**
     * Created: 2026-08-12T12:14:52+09:00
     * Modified:
     *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    /**
     * WHAT: 입력된 데이터 타입(dtype)이 프레임워크에서 지원하는 타입인지 검증하는 함수입니다.
     * WHY: 지원하지 않는 데이터 타입이 사용될 경우 발생할 수 있는 메모리 계산 오류 및 WebGPU 셰이더 오류를 사전에 방지하기 위해 존재합니다.
     * HOW: 입력된 dtype 문자열이 "float32"인지 비교하고, 일치하지 않으면 AMEVAForgeDTypeError 예외를 발생시킵니다. asserts 키워드를 사용하여 타입스크립트 컴파일러에게 dtype이 DType임을 보장합니다.
     */
    function validateDType(dtype) {
        // WHAT: dtype이 "float32"가 아닌지 확인하는 조건문입니다.
        // WHY: 현재 WebGPU 연산 파이프라인에서 float32 데이터 타입만 완벽하게 지원하므로 이를 검증하기 위함입니다.
        // HOW: 일치 연산자(!==)를 통해 입력 문자열이 정확히 "float32"와 다른지 확인합니다.
        if (dtype !== "float32") {
            throw new AMEVAForgeDTypeError(`Unsupported dtype: ${dtype}. Only float32 is supported.`);
        }
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * buffers.ts — GPU 버퍼 할당, 읽기 인터페이스
     *
     * C-05 Fix: _stagingBuffers 전역 Map 제거 → mapBufferAsync가 staging buffer 직접 반환.
     * H-05 / NH-05 Fix: "Zero-Copy" 주석 수정 — GPU→CPU 전송은 1번 copy가 불가피.
     *   WebGPU 스펙상 GPU 메모리를 WASM 힙과 직접 공유할 수 없다 (CUDA pinned memory와 달리).
     *   최소 1번의 copy는 WebGPU의 구조적 한계이며 Dawn, wgpu, TensorFlow.js도 동일.
     * ARC-01 Fix: createBuffer() OOM은 device.pushErrorScope()로만 감지 가능 — 문서화.
     */
    /**
     * WHAT: 지정된 크기와 용도에 맞게 GPU 버퍼를 할당합니다.
     * WHY: WebGPU의 버퍼 생성을 추상화하고 전역 할당량(Quota) 관리 시스템과 통합하여 메모리 부족(OOM)을 방지하기 위해 존재합니다.
     * HOW: QuotaManager를 통해 `byteLength`만큼의 메모리를 예약한 후, `device.createBuffer`를 호출하여 버퍼를 생성합니다. 실패 시 예약된 메모리 토큰을 반환(release)하고 에러를 던집니다.
     */
    function allocateBuffer(byteLength, usage, kind = 'tensor', ownerGraph = null) {
        const token = _globalQuotaManager.reserveToken(byteLength, kind, ownerGraph);
        try {
            const buffer = getDevice().createBuffer({ size: byteLength, usage });
            return { buffer, token };
        }
        catch (e) {
            _globalQuotaManager.releaseToken(token);
            throw e;
        }
    }
    /**
     * WHAT: 주어진 GPU 버퍼에 Float32Array 데이터를 씁니다.
     * WHY: CPU 측의 데이터를 GPU 버퍼로 복사하여 GPU 연산에 사용할 수 있도록 하기 위해 필요합니다.
     * HOW: WebGPU 큐(`device.queue.writeBuffer`)를 사용하여 주어진 데이터의 전체 크기만큼 지정된 버퍼의 오프셋 0부터 복사합니다.
     */
    function writeFloat32Array(buffer, data) {
        getQueue().writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    }
    /**
     * WHAT: GPU 버퍼의 데이터를 읽어서 CPU 메모리 상의 Float32Array로 반환합니다.
     * WHY: GPU에서 처리된 결과 데이터를 CPU로 가져와서 애플리케이션 수준에서 활용(예: 출력, 저장)하기 위해 존재합니다.
     * HOW:
     *   1. 복사를 위한 중간 버퍼(Staging Buffer)를 MAP_READ와 COPY_DST 용도로 할당합니다.
     *   2. CommandEncoder를 사용해 원본 버퍼의 데이터를 Staging Buffer로 복사하고 큐에 제출합니다.
     *   3. Staging Buffer를 비동기적으로 맵핑(mapAsync)하여 CPU에서 읽을 수 있게 합니다.
     *   4. 데이터를 읽어 Float32Array로 복사한 후 버퍼를 해제(unmap, destroy)하고 토큰을 반환합니다.
     */
    async function readBufferToFloat32Array(buffer, byteLength) {
        const device = getDevice();
        const { buffer: stagingBuffer, token } = allocateBuffer(byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'staging');
        try {
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
            device.queue.submit([commandEncoder.finish()]);
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            try {
                const arrayBuffer = stagingBuffer.getMappedRange();
                return new Float32Array(arrayBuffer.slice(0));
            }
            finally {
                stagingBuffer.unmap();
            }
        }
        finally {
            stagingBuffer.destroy();
            _globalQuotaManager.releaseToken(token);
        }
    }
    /**
     * WHAT: GPU 버퍼의 내용을 읽기 위해 Staging Buffer를 생성하고 비동기적으로 맵핑합니다.
     * WHY: 대용량 데이터 전송 시 메모리 맵핑을 직접 제어하거나 제로 카피(Zero-Copy) 메커니즘과 유사한 최적화를 구현하기 위해 필요합니다.
     * HOW: `MAP_READ | COPY_DST` 속성의 Staging 버퍼를 새로 할당하고, 원본 버퍼의 내용을 복사하기 위한 커맨드를 큐에 제출한 뒤, `mapAsync`를 호출하여 맵핑된 버퍼와 할당 토큰을 반환합니다.
     */
    async function mapBufferAsync$1(buffer, byteLength) {
        const device = getDevice();
        const { buffer: stagingBuffer, token } = allocateBuffer(byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'staging');
        try {
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, byteLength);
            device.queue.submit([commandEncoder.finish()]);
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            return { stagingBuffer, token };
        }
        catch (e) {
            stagingBuffer.destroy();
            _globalQuotaManager.releaseToken(token);
            throw e;
        }
    }
    /**
     * WHAT: 맵핑이 완료된 Staging 버퍼의 데이터를 외부에서 제공된 Float32Array 배열에 직접 복사합니다.
     * WHY: 새로운 배열 객체를 생성하지 않고 기존 메모리(Pre-allocated buffer)를 재사용하여 메모리 할당 및 가비지 컬렉션(GC) 부하를 줄이기 위해 사용됩니다.
     * HOW: Staging 버퍼의 맵핑 범위를 가져와서 전달된 `outArray`에 `set` 메서드로 데이터를 덮어쓴 후, 맵핑을 해제(unmap), 버퍼 파괴(destroy) 및 메모리 토큰을 해제합니다.
     */
    function readMappedInto$1(stagingBuffer, token, outArray) {
        try {
            const arrayBuffer = stagingBuffer.getMappedRange();
            const mapped = new Float32Array(arrayBuffer);
            if (outArray.length !== mapped.length) {
                throw new RangeError(`readMappedInto destination length mismatch: expected ${mapped.length}, got ${outArray.length}`);
            }
            outArray.set(mapped);
        }
        finally {
            stagingBuffer.unmap();
            stagingBuffer.destroy();
            _globalQuotaManager.releaseToken(token);
        }
    }
    /**
     * WHAT: 할당된 GPU 버퍼를 메모리에서 해제하고, 관련된 할당량 토큰(AllocationToken)을 반환합니다.
     * WHY: WebGPU 리소스 누수를 방지하고, 전역 쿼타 매니저(Quota Manager)에 반환하여 다른 작업에서 가용 메모리를 사용할 수 있도록 하기 위해 존재합니다.
     * HOW: `buffer.destroy()`를 호출하여 실제 GPU 리소스를 해제한 다음, `_globalQuotaManager.releaseToken(token)`을 통해 예약된 메모리 용량을 반환합니다.
     */
    function freeBuffer(buffer, token) {
        buffer.destroy();
        _globalQuotaManager.releaseToken(token);
    }

    /**
     * Created: 2026-08-12T12:14:52+09:00
     * Modified:
     *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * tensorRegistry.ts — GPU 텐서 생명주기 레지스트리
     *
     * C-06 Fix: dispose() 시 _globalQuotaManager.markPendingRelease() 즉시 호출.
     * NC-07 Fix: dynamic import() 제거 → 정적 import 사용 + device.destroy() 보장.
     * NL-03 Fix: Date.now() 제거 → 단조증가 ID만 사용 (타이밍 정보 노출 방지).
     */
    /**
     * WHAT: GPU 텐서의 생명주기를 관리하는 레지스트리 클래스입니다.
     * WHY: 생성된 텐서의 메타데이터와 WebGPU 버퍼를 중앙에서 추적하고 메모리 누수를 방지하기 위해 존재합니다.
     * HOW: Map 객체를 사용하여 고유한 핸들(TensorHandle)을 키로, 텐서 레코드(TensorRecord)를 값으로 저장 및 관리합니다.
     */
    class TensorRegistry {
        records = new Map();
        nextId = 1;
        snapshotHandles() {
            const list = [];
            for (const [handle, record] of this.records.entries()) {
                if (!record.disposed)
                    list.push(handle);
            }
            return list;
        }
        registerRecord(record) {
            console.warn(`[REGISTRY REGISTER_RECORD] handle=${record.handle}`);
            const fullRecord = {
                ...record,
                disposed: false,
                createdAt: this.nextId - 1,
            };
            this.records.set(record.handle, fullRecord);
            this.nextId++;
            return record.handle;
        }
        /**
         * WHAT: 새로운 텐서를 레지스트리에 등록하고 고유 핸들을 반환하는 함수입니다.
         * WHY: WebGPU 버퍼 및 메타데이터를 프레임워크가 추적할 수 있도록 레지스트리에 기록하기 위함입니다.
         * HOW: 예측 불가능한 UUID 기반의 핸들을 생성하고, 입력받은 정보와 함께 내부 records 맵에 저장합니다.
         */
        register(recordOmitHandle) {
            // F-015 Fix: 예측 가능한 핸들 생성을 막기 위해 암호학적 난수 기반 식별자 사용
            /**
             * WHAT: 암호학적으로 안전한 무작위 식별자(UUID) 문자열입니다.
             * WHY: 악의적인 사용자가 다른 텐서의 핸들을 추측하여 접근하는 것을 방지하기 위해 생성됩니다.
             * HOW: crypto.randomUUID가 사용 가능하면 이를 호출하고, 그렇지 않으면 Math.random()을 기반으로 임시 문자열을 생성합니다.
             */
            const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
                ? crypto.randomUUID()
                : Math.random().toString(36).substring(2, 15);
            /**
             * WHAT: 텐서를 고유하게 식별하기 위한 최종 핸들 문자열입니다.
             * WHY: 외부에서 텐서를 참조할 때 이 문자열을 사용하여 안전하게 접근할 수 있도록 제공됩니다.
             * HOW: "tensor_" 접두사와 위에서 생성한 uuid 문자열을 결합하여 생성됩니다.
             */
            const handle = `tensor_${uuid}`;
            /**
             * WHAT: 레지스트리에 저장될 텐서의 모든 메타데이터를 포함하는 레코드 객체입니다.
             * WHY: WebGPU 버퍼, 모양(shape), 데이터 타입, 생성 순서 등을 한 곳에서 관리하기 위함입니다.
             * HOW: 전달된 recordOmitHandle 객체에 handle, disposed=false, createdAt(단조증가 ID)을 병합하여 생성합니다.
             */
            const record = {
                ...recordOmitHandle,
                handle,
                disposed: false,
                createdAt: this.nextId - 1 // NL-03 Fix: monotonic counter, not timestamp
            };
            this.records.set(handle, record);
            this.nextId++;
            return handle;
        }
        /**
         * WHAT: 주어진 핸들을 사용하여 텐서 레코드를 조회하는 함수입니다.
         * WHY: 연산을 수행할 때 필요한 텐서의 메타데이터와 실제 WebGPU 버퍼를 가져오기 위해 존재합니다.
         * HOW: 내부 records 맵에서 핸들을 키로 조회하며, 존재하지 않거나 이미 폐기된 경우 에러를 발생시킵니다.
         */
        get(handle) {
            /**
             * WHAT: 레지스트리에서 핸들로 조회한 텐서 레코드입니다.
             * WHY: 텐서가 실제로 존재하는지 검증하고 접근하기 위해 임시 변수에 저장합니다.
             * HOW: this.records.get(handle)을 통해 값을 가져옵니다.
             */
            const record = this.records.get(handle);
            if (!record) {
                throw new AMEVAForgeDisposedError(`Tensor not found: ${handle}`);
            }
            if (record.disposed) {
                throw new AMEVAForgeDisposedError(`Attempted to access disposed tensor: ${handle}`);
            }
            return record;
        }
        /**
         * WHAT: 특정 핸들의 텐서가 유효하게 존재하는지 확인하는 함수입니다.
         * WHY: 텐서가 해제(dispose)되었는지 예외 없이 안전하게 체크하기 위해 사용됩니다.
         * HOW: 핸들로 레코드를 조회하여 undefined가 아니고 disposed 상태가 아닌지(boolean)를 반환합니다.
         */
        has(handle) {
            /**
             * WHAT: 조회된 텐서 레코드 변수입니다.
             * WHY: 존재 여부 및 disposed 상태를 판별하기 위해 사용합니다.
             * HOW: records 맵에서 핸들로 가져옵니다.
             */
            const record = this.records.get(handle);
            return record !== undefined && !record.disposed;
        }
        /**
         * WHAT: 지정된 핸들의 텐서를 폐기하고 GPU 메모리를 해제하는 함수입니다.
         * WHY: 사용이 끝난 텐서의 메모리를 반환하여 OOM(Out of Memory)을 방지하고 자원 누수를 막기 위함입니다.
         * HOW: 레코드를 disposed로 표시하고 맵에서 제거한 뒤, QuotaManager와 WebGPU 큐를 통해 버퍼를 해제합니다.
         */
        dispose(handle) {
            console.warn(`[REGISTRY DISPOSE CALLED] handle=${handle}`);
            if (!this.records.has(handle)) {
                return; // TS-H04: 이중 dispose 방어 — 이미 해제된 핸들 무시
            }
            const record = this.records.get(handle);
            if (!record || record.disposed)
                return;
            record.disposed = true;
            this.records.delete(handle);
            // C-06 Fix: 즉시 "해제 예약" 표시
            _globalQuotaManager.markPendingRelease(record.token);
            // NC-07 Fix: 정적 import된 getDevice() 사용 (dynamic import 제거)
            try {
                /**
                 * WHAT: 현재 활성화된 WebGPU 디바이스 인스턴스입니다.
                 * WHY: GPU에 제출된 모든 명령이 끝난 후 안전하게 버퍼를 파괴하기 위해 필요합니다.
                 * HOW: getDevice() 유틸리티 함수를 호출하여 가져옵니다.
                 */
                const device = getDevice();
                device.queue.onSubmittedWorkDone().then(() => {
                    freeBuffer(record.buffer, record.token);
                }).catch(() => {
                    // GPU 큐 실패 → 즉시 소각
                    _safeDestroyBuffer(record);
                });
            }
            catch {
                // device가 없거나 lost → 즉시 quota 해제 + buffer 소각
                _safeDestroyBuffer(record);
            }
        }
        // F-016 Fix: 비동기 에러 발생 시 해당 핸들에 에러를 마킹
        markFailed(handle, errorMsg) {
            const record = this.records.get(handle);
            if (record && !record.disposed) {
                record.error = errorMsg;
            }
        }
        /**
         * WHAT: 레지스트리에 등록된 모든 텐서를 일괄 폐기하는 함수입니다.
         * WHY: 컨텍스트 초기화나 애플리케이션 종료 시 모든 GPU 자원을 확실하게 정리하기 위해 존재합니다.
         * HOW: 아직 폐기되지 않은 모든 레코드를 수집하고, 맵을 비운 뒤 GPU 큐가 비워지면 버퍼를 순차적으로 해제합니다.
         */
        clear() {
            /**
             * WHAT: 아직 해제되지 않아 메모리를 점유하고 있는 텐서 레코드들의 배열입니다.
             * WHY: 맵(Map)이 초기화된 후에도 이 객체들의 버퍼를 파괴하기 위해 참조를 유지해야 합니다.
             * HOW: this.records.values()를 배열로 변환하고 disposed가 false인 것만 필터링합니다.
             */
            const recordsToFree = Array.from(this.records.values()).filter(r => !r.disposed);
            this.records.clear();
            if (recordsToFree.length === 0)
                return;
            /**
             * WHAT: 해제 대상 텐서들을 순회하는 루프입니다.
             * WHY: 모든 할당된 텐서에 대해 해제 대기 상태임을 QuotaManager에 알리기 위함입니다.
             * HOW: for...of 구문을 사용하여 recordsToFree 배열을 순회합니다.
             */
            for (const record of recordsToFree) {
                _globalQuotaManager.markPendingRelease(record.token);
            }
            try {
                /**
                 * WHAT: WebGPU 명령 큐의 상태를 확인하기 위한 디바이스 객체입니다.
                 * WHY: 큐에 대기 중인 작업이 텐서를 참조할 수 있으므로, 작업 완료 후 안전하게 해제하기 위해 사용됩니다.
                 * HOW: getDevice()를 통해 인스턴스를 얻어옵니다.
                 */
                const device = getDevice();
                device.queue.onSubmittedWorkDone().then(() => {
                    /**
                     * WHAT: GPU 작업이 완료된 후 각 버퍼를 해제하는 루프입니다.
                     * WHY: 실제 VRAM과 QuotaManager의 할당량을 반환하기 위해 필요합니다.
                     * HOW: for...of 루프를 돌며 freeBuffer를 호출합니다.
                     */
                    for (const record of recordsToFree) {
                        freeBuffer(record.buffer, record.token);
                    }
                }).catch(() => {
                    /**
                     * WHAT: GPU 큐 대기 실패 시 강제 해제하는 루프입니다.
                     * WHY: 큐에러가 발생해도 메모리 누수를 방지하기 위해 존재합니다.
                     * HOW: _safeDestroyBuffer 헬퍼를 직접 호출합니다.
                     */
                    for (const record of recordsToFree) {
                        _safeDestroyBuffer(record);
                    }
                });
            }
            catch {
                // device already lost
                /**
                 * WHAT: 디바이스 유실 시 텐서 버퍼를 강제 해제하는 루프입니다.
                 * WHY: 디바이스가 유실되어 큐 대기를 할 수 없으므로 남은 리소스를 정리하기 위해 필요합니다.
                 * HOW: for...of 루프를 돌며 _safeDestroyBuffer를 호출하고, 이후 쿼터를 초기화합니다.
                 */
                for (const record of recordsToFree) {
                    _safeDestroyBuffer(record);
                }
                _globalQuotaManager.reset();
            }
        }
    }
    /**
     * WHAT: 텐서의 WebGPU 버퍼를 파괴하고 할당 토큰을 해제하는 유틸리티 함수입니다.
     * WHY: 디바이스 유실이나 큐 실패 상황에서도 예외 발생 없이 버퍼 자원을 반환하기 위해 존재합니다.
     * HOW: try-catch 블록 안에서 buffer.destroy()를 호출하고, _globalQuotaManager.releaseToken()을 호출합니다.
     */
    function _safeDestroyBuffer(record) {
        try {
            record.buffer.destroy();
        }
        catch {
            // buffer가 이미 destroyed
        }
        _globalQuotaManager.releaseToken(record.token);
    }
    /**
     * WHAT: 전역적으로 사용되는 텐서 레지스트리의 단일 인스턴스(싱글톤)입니다.
     * WHY: 애플리케이션 전체에서 동일한 텐서 관리 상태를 공유하기 위해 생성됩니다.
     * HOW: TensorRegistry 클래스의 새 인스턴스를 생성하여 내보냅니다(export).
     */
    const _globalRegistry = new TensorRegistry();

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * pipelineCache.ts — WGSL 컴파일 파이프라인 캐시
     *
     * L-03 Fix: clear() 메서드를 통해 device lost 시 캐시 무효화.
     * NL-02 Fix: 캐시 키에 WGSL 해시를 포함하여 동일 op명으로 다른 WGSL 지원.
     */
    /**
     * WHAT: 문자열 데이터를 기반으로 고유한 해시(32-bit 정수형 기반 16진수 문자열)를 생성하는 간단한 해시 함수입니다.
     * WHY: WGSL 소스 코드 문자열을 해시화하여 파이프라인 캐시 키(cache key)에 추가함으로써, 같은 이름의 연산(op)이라도 다른 WGSL 코드가 주어질 경우 충돌을 방지하기 위함입니다.
     * HOW: djb2 해시 알고리즘 변형을 사용합니다.
     *      초기값 5381에서 시작하여 문자열의 각 문자 코드를 순회(for 루프)하면서,
     *      비트 이동 및 XOR 연산을 통해 해시 값을 누적한 후 부호 없는 32비트 정수를 16진수 문자열로 변환하여 반환합니다.
     */
    function hashString(str) {
        let hash = 5381;
        /**
         * WHAT: 문자열의 각 문자를 순회하며 해시값을 갱신하는 반복문입니다.
         * WHY: 문자열 전체의 데이터를 기반으로 고유성을 보장하는 해시값을 계산하기 위해 필요합니다.
         * HOW: 인덱스 i를 0부터 문자열 끝까지 증가시키며 현재 문자의 유니코드 값을 가져와 비트 연산으로 기존 해시값과 혼합합니다.
         */
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
            hash = hash & hash; // 32bit integer
        }
        return (hash >>> 0).toString(16);
    }
    /**
     * WHAT: WebGPU 컴퓨트 파이프라인(GPUComputePipeline)과 셰이더 모듈(GPUShaderModule)을 저장하고 재사용하는 캐시 관리 클래스입니다.
     * WHY: WGSL 코드를 매번 파싱하고 컴파일하는 비용(오버헤드)을 줄여 GPU 연산 초기화 성능을 극대화하기 위해 존재합니다.
     * HOW: 내부적으로 Map 인스턴스를 유지하여 연산명(key)과 WGSL 해시의 조합을 캐시 키로 사용하고, 컴파일된 객체를 메모리에 저장 및 반환합니다.
     */
    class PipelineCache {
        /**
         * WHAT: 컴파일 완료된 셰이더 모듈과 컴퓨트 파이프라인 객체를 키(문자열)에 매핑하여 보관하는 내부 저장소입니다.
         * WHY: 반복적인 연산 요청 시 동일한 코드가 주어지면 이전에 컴파일된 객체를 빠르게 찾아 반환하기 위해 필요합니다.
         * HOW: JavaScript 내장 Map 구조를 사용하여 생성되며, 캐시 적중(Cache Hit) 시 저장된 값을 제공하고, 누락(Cache Miss) 시 새 객체를 추가합니다.
         */
        cache = new Map();
        /**
         * WHAT: 주어진 연산 이름(key)과 WGSL 소스 코드를 바탕으로 컴파일된 파이프라인 객체와 셰이더 모듈을 반환합니다.
         * WHY: 기존에 컴파일된 캐시가 있다면 즉시 반환하여 성능을 최적화하고, 없다면 즉석에서(Synchronously) 새로 컴파일하기 위해 사용됩니다.
         * HOW: 연산 이름과 WGSL 해시를 조합해 캐시 키를 만든 후, 내부 캐시 맵에서 조회합니다. 없을 경우 WebGPU 디바이스에 셰이더 모듈과 파이프라인을 생성 요청하고, 결과를 캐시에 저장한 뒤 반환합니다.
         */
        getPipeline(key, wgslCode) {
            const cacheKey = `${key}:${hashString(wgslCode)}`;
            const cached = this.cache.get(cacheKey);
            if (cached)
                return cached;
            const device = getDevice();
            const shader = device.createShaderModule({ code: wgslCode });
            const pipeline = device.createComputePipeline({
                layout: "auto",
                compute: { module: shader, entryPoint: "main" },
            });
            const entry = { shader, pipeline };
            this.cache.set(cacheKey, entry);
            return entry;
        }
        /**
         * WHAT: 애플리케이션 초기화 단계에서 지정된 여러 파이프라인을 비동기적으로 미리 컴파일하여 캐싱하는 웜업(Warmup) 기능입니다.
         * WHY: 첫 번째 GPU 연산 실행 시 발생하는 동기적 컴파일로 인한 UI 프리징(Freeze) 혹은 끊김(Stuttering) 현상을 방지하기 위해 존재합니다.
         * HOW: 입력받은 배열을 순회하며 아직 캐시되지 않은 항목만 추려낸 뒤, `createComputePipelineAsync`를 사용해 비동기로 병렬 컴파일을 수행(Promise.allSettled)하고 결과를 캐시에 저장합니다.
         */
        async warmup(entries) {
            const device = getDevice();
            const pendingEntries = entries.filter(e => !this.cache.has(`${e.key}:${hashString(e.wgslCode)}`));
            const promises = pendingEntries.map(async (e) => {
                const cacheKey = `${e.key}:${hashString(e.wgslCode)}`;
                const shader = device.createShaderModule({ code: e.wgslCode });
                const pipeline = await device.createComputePipelineAsync({
                    layout: "auto",
                    compute: { module: shader, entryPoint: "main" },
                });
                this.cache.set(cacheKey, { shader, pipeline });
            });
            const results = await Promise.allSettled(promises);
            results.forEach((res, i) => {
                if (res.status === 'rejected') {
                    console.warn(`[AMEVA] Warmup failed for ${pendingEntries[i].key}:`, res.reason);
                }
            });
        }
        /**
         * WHAT: 파이프라인 캐시 내에 저장된 모든 항목을 삭제하여 완전히 초기화합니다.
         * WHY: GPU 디바이스가 유실(device lost)되었거나 초기화가 다시 발생할 때, 이전 디바이스 컨텍스트를 가리키는 더 이상 유효하지 않은 파이프라인 참조를 제거하기 위함입니다.
         * HOW: 내부 맵(Map) 객체의 내장 메서드인 `clear()`를 호출하여 모든 키-값 쌍을 비웁니다.
         */
        clear() {
            this.cache.clear();
        }
        /**
         * WHAT: 현재 캐시에 저장된 파이프라인 객체의 총 개수를 반환하는 프로퍼티 접근자(Getter)입니다.
         * WHY: 메모리 사용량 모니터링이나 디버깅 시 캐시의 누적 상태를 파악하기 위해 제공됩니다.
         * HOW: 내부 캐시 맵의 `size` 프로퍼티 값을 그대로 반환합니다.
         */
        get size() {
            return this.cache.size;
        }
    }
    /**
     * WHAT: 전역에서 공유되는 단일 PipelineCache 인스턴스입니다.
     * WHY: 여러 텐서 연산 모듈이 개별 캐시를 만들지 않고 하나의 중앙 집중형 캐시를 재사용하여 메모리와 컴파일 비용을 최소화하기 위해 사용됩니다.
     * HOW: PipelineCache 클래스의 인스턴스를 하나 생성하여 모듈 외부로 노출(export)합니다.
     */
    const _globalPipelineCache = new PipelineCache();

    /**
     * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     * 수정 이력:
     * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const MATMUL_WGSL = `
/**
 * 이 구조체(Params)는 행렬 곱셈 연산(A x B = C)에 필요한 차원 정보와 오프셋을 제공하기 위해 존재합니다.
 * A행렬은 (M x K), B행렬은 (K x N), C행렬은 (M x N) 차원을 가집니다.
 */
struct Params {
  M: u32, // 행렬 A와 C의 행(Row) 개수입니다.
  N: u32, // 행렬 B와 C의 열(Column) 개수입니다.
  K: u32, // 행렬 A의 열(Column) 개수이자 행렬 B의 행(Row) 개수입니다 (내적을 수행할 길이).
  offsetY: u32, // 워크그룹 파견 한계(dispatch limit)를 우회하기 위해 y축 시작 오프셋을 지정합니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 행렬의 형태 및 오프셋 정보를 담은 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> a: array<f32>; // (M x K) 크기의 첫 번째 입력 행렬 데이터입니다.
@group(0) @binding(2) var<storage, read> b: array<f32>; // (K x N) 크기의 두 번째 입력 행렬 데이터입니다.
@group(0) @binding(3) var<storage, read_write> c: array<f32>; // 결과값(M x N)이 기록될 출력 행렬 데이터입니다.

/**
 * main 함수는 두 행렬 A와 B를 곱하여 결과 행렬 C를 계산합니다.
 * 딥러닝에서 가장 핵심적인 연산인 GEMM(General Matrix Multiply)을 GPU로 분산하여 병렬 처리하기 위해 존재합니다.
 */
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // global_id.z 를 X축 타일 오프셋으로 사용
  // dispatcher가 z = ceil(N / (65535*8))만큼 dispatch
  
  // 현재 스레드가 계산을 담당할 출력 행렬 C의 열(Column) 인덱스를 계산합니다.
  // z축 워크그룹 인덱스를 사용하여 1D 한계를 넘는 큰 행렬에 대한 스팬(span)을 지원합니다.
  let col = global_id.x + global_id.z * 65535u * 8u;
  // 현재 스레드가 계산을 담당할 출력 행렬 C의 행(Row) 인덱스를 계산합니다 (오프셋 포함).
  let row = global_id.y + params.offsetY;

  // 계산된 인덱스가 행렬 C의 범위를 초과하는 스레드는 작업을 수행하지 않고 바로 종료합니다.
  if (row >= params.M || col >= params.N) {
    return;
  }

  // A행렬의 row번째 행과 B행렬의 col번째 열 사이의 내적(Dot product)을 누적할 변수입니다.
  var sum: f32 = 0.0;
  
  // 내적을 수행하기 위해 공통 차원인 K번 만큼 루프를 돕니다.
  // A의 원소와 B의 원소를 순차적으로 곱하여 합산합니다.
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    sum = sum + a[row * params.K + k] * b[k * params.N + col];
  }

  // 계산된 최종 내적 값을 결과 행렬 C의 해당하는 1D 인덱스 위치에 저장합니다.
  c[row * params.N + col] = sum;
}
`;

    /**
     * 생성일: 2026-08-12T12:14:52+09:00
     * 수정 이력:
     * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const RELU_WGSL = `
// 구조체: Params
// 역할 (WHAT): ReLU(Rectified Linear Unit) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼(uniform) 데이터를 효율적으로 전달하고 메모리 정렬을 맞추기 위해 사용됩니다.
// 동작 방식 (HOW): 각 스레드가 처리할 전체 요소 개수와 2D 워크그룹 할당 정보를 메모리에서 읽어옵니다.
struct Params {
  // 변수: size
  // 역할: 입력 텐서가 가진 총 데이터 요소의 수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향으로 할당된 작업 그룹(workgroup)의 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 정렬을 맞추기 위한 패딩 변수
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체 값을 담고 있는 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: ReLU 활성화 함수가 적용될 원본 데이터를 가진 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: ReLU 연산 결과가 기록될 읽기/쓰기 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 각 요소에 대해 ReLU 활성화 함수(max(0, x))를 적용합니다.
// 목적 (WHY): 딥러닝 모델의 비선형성을 부여하기 위한 ReLU 연산을 GPU에서 병렬로 고속 처리하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 전역 ID를 활용하여 자신의 1D 인덱스를 계산한 후, 해당 인덱스에 있는 x의 값과 0 중 더 큰 값을 y에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 처리할 배열의 전체 요소 수를 유니폼 버퍼로부터 가져옵니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 글로벌 ID의 2D 인덱스를 1D 인덱스로 변환하기 위해 X축 워크그룹 수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: 현재 스레드가 처리해야 하는 1차원 데이터의 절대 인덱스
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 배열 크기 검사
  // 역할: 계산된 인덱스가 실제 데이터 범위(num_elements)를 벗어나는 경우 쓰레드 실행을 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 해당 위치 값을 0.0과 비교해 큰 값(음수는 0, 양수는 그대로)을 y 배열에 저장합니다.
  y[idx] = max(x[idx], 0.0);
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:14:52 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const ADD_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서의 덧셈 연산을 수행할 때 필요한 파라미터들을 담고 있는 구조체입니다.
 * GPU 내에서 uniform 버퍼를 통해 전달받아 연산의 크기나 차원을 제어하는 목적(Why)으로 사용됩니다.
 */
struct Params {
  // 전체 요소(element)의 개수입니다. 배열 범위를 초과하지 않도록 경계 검사를 수행하는 데(How) 사용됩니다.
  size: u32,
  // X 차원의 워크그룹 개수입니다. 2D 이상의 그리드에서 1차원 인덱스를 계산하기 위해(What) 필요합니다.
  workgroups_x: u32,
  size_a: u32,
  size_b: u32,
};

// params: GPU와 CPU 간의 데이터를 동기화하기 위한 Uniform 버퍼입니다. 연산에 필요한 메타데이터가 담겨 있습니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 첫 번째 입력 텐서의 데이터가 저장된 배열입니다. 읽기 전용(storage, read)으로 선언되었습니다.
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 두 번째 입력 텐서의 데이터가 저장된 배열입니다. 읽기 전용(storage, read)으로 선언되었습니다.
@group(0) @binding(2) var<storage, read> b: array<f32>;
// out: 연산 결과(덧셈)가 저장되는 출력 배열입니다. 읽고 쓰기가 가능한(storage, read_write) 형태로 선언되었습니다.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    let idx_a = select(idx % params.size_a, 0u, params.size_a == 1u);
    let idx_b = select(idx % params.size_b, 0u, params.size_b == 1u);
    out[idx] = a[idx_a] + b[idx_b];
  }
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:59:35: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization) (67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const TRANSPOSE_WGSL = `
// 구조체: Params
// 목적: 행렬 전치(Transpose) 연산에 필요한 차원 정보를 전달합니다.
// 작동 방식: 행(M), 열(N), 배치(B) 크기를 받아 다차원 배열의 인덱스를 계산할 수 있게 합니다.
struct Params {
  // 변수: M
  // 목적: 변환 전 원본 행렬의 행(Row) 개수입니다.
  // 작동 방식: 전치 후에는 이 값이 열의 개수가 됩니다.
  M: u32,
  // 변수: N
  // 목적: 변환 전 원본 행렬의 열(Column) 개수입니다.
  // 작동 방식: 전치 후에는 이 값이 행의 개수가 됩니다.
  N: u32,
  // 변수: B
  // 목적: 배치(Batch) 크기를 의미합니다.
  // 작동 방식: 여러 개의 독립적인 행렬(배치)을 동시에 전치할 수 있게 합니다.
  B: u32,
};

// 변수: params
// 목적: 셰이더 실행 시 필요한 차원(M, N, B) 정보를 담은 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0에 매핑되어 인덱스 계산의 기준값으로 사용됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 전치하기 전의 원본 다차원 배열(배치 포함 3차원 구조) 데이터입니다.
// 작동 방식: 바인딩 1에 읽기 전용 스토리지 버퍼로 바인딩됩니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: out
// 목적: 행과 열이 뒤바뀐 전치 결과를 저장할 출력 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되어 계산된 데이터가 저장됩니다.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: 배치 차원을 유지한 채로 행(Row)과 열(Column)의 위치를 바꾸는 전치 연산을 수행합니다.
// 작동 방식: 3차원 글로벌 인덱스(x, y, z)를 각각 (row, col, batch)로 매핑하고 변환 공식을 적용합니다.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: row
  // 목적: 원본 행렬 기준에서의 행 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 x 성분(global_id.x)을 사용합니다.
  let row = global_id.x;

  // 변수: col
  // 목적: 원본 행렬 기준에서의 열 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 y 성분(global_id.y)을 사용합니다.
  let col = global_id.y;

  // 변수: batch
  // 목적: 현재 처리 중인 배치의 인덱스를 나타냅니다.
  // 작동 방식: 3D 스레드 ID의 z 성분(global_id.z)을 사용합니다.
  let batch = global_id.z;
  
  // 제어문: if
  // 목적: 패딩이나 워크그룹 크기 맞춤으로 인해 실제 데이터 범위를 초과한 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: row, col, batch가 각각 M, N, B보다 작은지 확인합니다.
  if (row < params.M && col < params.N && batch < params.B) {
    // 변수: in_idx
    // 목적: 1차원으로 평면화된 원본 배열에서 읽어올 요소의 인덱스를 계산합니다.
    // 작동 방식: '배치 오프셋 + 행 오프셋 + 열' (batch * M * N + row * N + col) 공식을 사용합니다.
    let in_idx = batch * (params.M * params.N) + row * params.N + col;

    // 변수: out_idx
    // 목적: 전치된 결과를 저장할 출력 배열의 1차원 평면화 인덱스를 계산합니다.
    // 작동 방식: 행과 열의 기준 크기가 바뀌므로 '배치 오프셋 + 새로운 행 오프셋 + 새로운 열' (batch * M * N + col * M + row)로 계산합니다.
    let out_idx = batch * (params.M * params.N) + col * params.M + row;

    // 연산: out[out_idx] 할당
    // 목적: 계산된 위치에 원본 데이터를 복사하여 전치를 완료합니다.
    // 작동 방식: 원본 위치(in_idx)의 값을 읽어 목표 위치(out_idx)에 기록합니다.
    out[out_idx] = input[in_idx];
  }
}
`;

    /**
     * 생성일: 2026-08-12T12:14:52+09:00
     * 수정 이력:
     * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const MUL_WGSL = `
// 구조체: Params
// 역할 (WHAT): 곱셈 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼(uniform) 데이터를 효율적으로 전달하기 위해 존재합니다.
// 동작 방식 (HOW): 각 스레드가 처리해야 할 전체 크기 및 작업 그룹 설정 값을 메모리에서 읽어옵니다.
struct Params {
  size: u32,
  workgroups_x: u32,
  size_a: u32,
  size_b: u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> A : array<f32>;
@group(0) @binding(2) var<storage, read> B : array<f32>;
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  if (index < num_elements) {
    let idx_a = select(index % params.size_a, 0u, params.size_a == 1u);
    let idx_b = select(index % params.size_b, 0u, params.size_b == 1u);
    C[index] = A[idx_a] * B[idx_b];
  }
}
`;

    /**
     * 생성일: 2026-08-12T12:14:52+09:00
     * 수정 이력:
     * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const RELU_BACKWARD_WGSL = `
// 구조체: Params
// 역할 (WHAT): ReLU 역전파 연산에 필요한 메타데이터 정보를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 데이터를 전달하여 전체 연산 크기 등을 파악하게 합니다.
// 동작 방식 (HOW): 전체 크기(size)와 2D 그리드 변환을 위한 workgroups_x 인자를 넘겨줍니다.
struct Params {
  // 변수: size
  // 역할: 처리할 데이터의 전체 요소 수
  size: u32,
  // 변수: workgroups_x
  // 역할: x축 워크그룹의 총 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 메모리 정렬을 위한 패딩
  pad2: u32,
  pad3: u32,
}

// 변수: params
// 역할: 연산에 필요한 메타데이터가 담긴 유니폼 버퍼
@group(0) @binding(0) var<uniform> params : Params;

// 변수: X
// 역할: 순전파 시 입력되었던 원본 데이터를 담은 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> X : array<f32>;

// 변수: gradOutput
// 역할: 이전 레이어에서 흘러들어온 그래디언트(Gradient) 값을 담은 읽기 전용 버퍼
@group(0) @binding(2) var<storage, read> gradOutput : array<f32>;

// 변수: gradInput
// 역할: ReLU 연산의 역전파 결과로 계산된 그래디언트를 저장할 읽기/쓰기 버퍼
@group(0) @binding(3) var<storage, read_write> gradInput : array<f32>;

// 함수: main
// 역할 (WHAT): ReLU 역전파 그래디언트를 계산하는 컴퓨트 셰이더 메인 함수입니다.
// 목적 (WHY): 역전파 과정에서 X > 0 인 위치에만 그래디언트를 통과시키기 위해 존재합니다.
// 동작 방식 (HOW): 각 스레드는 1D 인덱스를 계산하고, X[index]의 값이 양수일 경우 gradOutput을 그대로 gradInput에 복사하고, 0 이하일 경우 0.0을 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // 변수: num_elements
  // 역할: 처리할 배열 요소의 총 개수를 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2D 인덱스를 1D 인덱스로 풀기 위해 가로 워크그룹 크기를 저장합니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: index
  // 역할: 현재 스레드의 작업을 가리키는 1차원 배열 위치 인덱스
  let index = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 데이터 경계 확인
  // 역할: 유효한 데이터 인덱스(num_elements 내부)인 경우에만 연산을 수행합니다.
  if (index < num_elements) {
    // 조건문: ReLU 미분 조건 (X > 0)
    // 역할: 원본 입력 값(X)이 양수인지 판단합니다.
    if (X[index] > 0.0) {
      // 변수 gradInput 갱신 (통과)
      // 역할: X가 양수였으므로 미분값이 1이 되어, 들어온 그래디언트를 그대로 전달합니다.
      gradInput[index] = gradOutput[index];
    } else {
      // 변수 gradInput 갱신 (차단)
      // 역할: X가 0 이하이므로 미분값이 0이 되어, 그래디언트 흐름을 0으로 차단합니다.
      gradInput[index] = 0.0;
    }
  }
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const SUB_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터(Uniform parameters)들을 정의합니다.
// 작동 방식: 배열의 크기(size)와 2차원 워크그룹 배열의 X축 크기(workgroups_x)를 제공합니다.
struct Params {
  size: u32,
  workgroups_x: u32,
  size_a: u32,
  size_b: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    let idx_a = select(idx % params.size_a, 0u, params.size_a == 1u);
    let idx_b = select(idx % params.size_b, 0u, params.size_b == 1u);
    out[idx] = a[idx_a] - b[idx_b];
  }
}
`;

    /**
     * 생성일: 2026-08-12T12:14:52+09:00
     * 수정 이력:
     * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const NEG_WGSL = `
// 구조체: Params
// 역할 (WHAT): 음수화(Negation) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 데이터를 전달하고 16바이트 정렬 규칙을 준수하기 위해 정의되었습니다.
// 동작 방식 (HOW): 셰이더가 실행될 때 전체 데이터 크기(size)와 2D 기반 분할 시 사용되는 x축 워크그룹 크기를 참조합니다.
struct Params {
  // 변수: size
  // 역할: 처리할 전체 데이터 배열의 요소 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 작업 그룹 수 (2D 그리드 인덱싱에 사용)
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 구조체의 메모리 정렬을 위한 여분(padding) 공간
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체를 저장하는 유니폼 버퍼 (인덱스 바인딩 0)
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: 입력 데이터를 담는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: 계산 결과(음수화된 값)가 저장될 읽기/쓰기 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 모든 요소에 대해 부호를 반전시키는 메인 컴퓨트 함수입니다.
// 목적 (WHY): GPU의 병렬 아키텍처를 활용하여 빠르고 동시적인 부호 반전 연산을 수행하기 위함입니다.
// 동작 방식 (HOW): 64 워크그룹 사이즈 내에서 각 스레드가 전역 ID를 사용해 1D 인덱스를 계산하고, 유효한 범위 내에서 - 연산을 적용합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 인자를 통해 전달된 전체 배열 크기를 저장합니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2D 그리드 맵핑을 풀기 위한 가로(X축) 워크그룹의 개수를 저장합니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: x 및 y 방향 워크그룹 ID와 로컬 ID를 결합하여 처리할 1D 데이터 인덱스를 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: out-of-bounds 방지
  // 역할: 계산된 인덱스가 실제 데이터의 요소 개수를 초과하면 처리를 조기 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 해당 인덱스 값을 읽어와 음수 기호를 붙인 후 y 배열에 씁니다.
  y[idx] = -x[idx];
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:14:52 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const DIV_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서 간의 요소별 나눗셈(division) 연산을 수행할 때 필요한 파라미터입니다. (What)
 * 셰이더 내에서 배열 크기 경계를 확인하고 3D 스레드 인덱스를 1차원으로 풀기 위해 사용됩니다. (Why)
 */
struct Params {
  size: u32,
  workgroups_x: u32,
  size_a: u32,
  size_b: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size;
  let workgroups_x = params.workgroups_x;
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  if (idx < num_elements) {
    let idx_a = select(idx % params.size_a, 0u, params.size_a == 1u);
    let idx_b = select(idx % params.size_b, 0u, params.size_b == 1u);
    let divisor = b[idx_b];
    let safe_divisor = select(divisor, select(1e-7, -1e-7, divisor < 0.0), abs(divisor) < 1e-7);
    out[idx] = a[idx_a] / safe_divisor;
  }
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     * 수정 이력:
     * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const EXP_WGSL = `
/**
 * 이 구조체(Params)는 워크그룹과 데이터의 크기를 설정하기 위해 존재합니다.
 * 패딩 변수들은 WebGPU 버퍼의 16바이트 정렬 규칙을 준수하기 위해 사용됩니다.
 */
struct Params {
  size: u32, // 처리해야 할 전체 요소의 개수입니다.
  workgroups_x: u32, // X축 방향으로 스패닝된 워크그룹의 총 개수입니다.
  pad2: u32, // 메모리 정렬을 위해 존재하는 사용되지 않는 패딩 변수입니다.
  pad3: u32, // 메모리 정렬을 위해 존재하는 사용되지 않는 패딩 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 상수 파라미터입니다.
@group(0) @binding(1) var<storage, read> x: array<f32>; // 읽기 전용으로 설정된 입력 텐서 데이터입니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // 연산 결과가 쓰여질 출력 텐서 데이터입니다.

/**
 * main 함수는 각 텐서 요소에 대해 자연 상수 e를 밑으로 하는 지수 함수(exp) 연산을 수행합니다.
 * 이 함수가 존재하는 이유는 텐서의 모든 원소에 대해 병렬적으로 지수 연산을 처리하기 위함입니다.
 * GPU의 각 스레드는 고유한 global_id를 받아 배열 내 자신의 작업 위치를 계산하고 결과를 출력 버퍼에 저장합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size; // 전체 계산해야 하는 원소의 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x; // 3D 그리드 기반의 1D 인덱스 계산을 위해 x축 워크그룹 수를 가져옵니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u; // 2D 형태로 스패닝된 글로벌 ID를 1D 인덱스로 변환하여 현재 스레드가 처리할 데이터의 위치를 구합니다.
  
  // 현재 스레드의 인덱스가 전체 배열 크기를 초과하면, 더 이상 처리하지 않고 함수를 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 계산된 인덱스의 입력값 x[idx]에 대해 지수 함수를 적용한 뒤, 그 결과를 출력 배열 y의 동일한 위치에 저장합니다.
  y[idx] = exp(x[idx]);
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     * 수정 이력:
     * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const LOG_WGSL = `
/**
 * 이 구조체(Params)는 워크그룹과 데이터의 크기를 설정하기 위해 존재합니다.
 * 패딩 변수들은 WebGPU 버퍼의 16바이트 정렬 규칙을 준수하기 위해 사용됩니다.
 */
struct Params {
  size: u32, // 처리해야 할 텐서의 전체 원소 개수입니다.
  workgroups_x: u32, // X축을 따라 생성된 워크그룹의 총 개수입니다.
  pad2: u32, // 16바이트 메모리 정렬을 위해 남겨둔 미사용 변수입니다.
  pad3: u32, // 16바이트 메모리 정렬을 위해 남겨둔 미사용 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 메타데이터 및 설정값이 담긴 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> x: array<f32>; // 자연로그 연산을 수행할 대상이 되는 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // 자연로그 연산 결과가 저장될 출력 텐서입니다.

/**
 * main 함수는 입력 텐서의 각 요소에 대하여 자연로그(log) 연산을 수행합니다.
 * 요소별(element-wise) 자연로그 연산을 GPU의 병렬 처리 능력을 통해 가속화하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.size; // 텐서의 전체 원소 개수를 유니폼 변수로부터 가져옵니다.
  let workgroups_x = params.workgroups_x; // 1차원 인덱스로 변환하기 위해 X축 워크그룹 크기를 가져옵니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u; // 2D 형태의 global_id를 1차원 평면 인덱스로 펼쳐서 현재 스레드의 작업 위치를 결정합니다.
  
  // 계산된 현재 스레드의 인덱스가 전체 텐서 크기를 벗어나면 작업을 수행하지 않고 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 현재 인덱스에 해당하는 입력 텐서 값에 대해 내장 함수 log()를 호출하고, 그 결과를 출력 텐서의 동일 위치에 저장합니다.
  y[idx] = log(x[idx]);
}
`;

    /**
     * 생성일: 2026-08-12T12:14:52+09:00
     * 수정 이력:
     * - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const SIGMOID_WGSL = `
// 구조체: Params
// 역할 (WHAT): 시그모이드(Sigmoid) 연산에 필요한 메타데이터를 저장하는 구조체입니다.
// 목적 (WHY): WebGPU 컴퓨트 셰이더로 유니폼 인자를 넘겨주어 전체 요소 수 등의 전역 설정을 공유하기 위함입니다.
// 동작 방식 (HOW): 요소 크기와 2D 워크그룹 할당 정보를 메모리 정렬을 맞추어 전달합니다.
struct Params {
  // 변수: size
  // 역할: 처리 대상 배열이 가진 전체 원소의 개수
  size: u32,
  // 변수: workgroups_x
  // 역할: X축 방향의 워크그룹 개수
  workgroups_x: u32,
  // 변수: pad2, pad3
  // 역할: 16바이트 메모리 정렬(alignment)용 패딩
  pad2: u32,
  pad3: u32,
};

// 변수: params
// 역할: Params 구조체를 담고 있는 유니폼 버퍼 (바인딩 0)
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 역할: 시그모이드 활성화 함수가 적용될 원본 데이터가 저장된 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 역할: 시그모이드 연산 결과가 기록될 읽기/쓰기 가능한 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 역할 (WHAT): 입력 텐서의 각 요소에 대해 시그모이드 활성화 함수(1 / (1 + exp(-x)))를 적용합니다.
// 목적 (WHY): 신경망의 값을 0과 1 사이로 변환하는 활성화 함수 연산을 GPU에서 병렬로 고속 수행하기 위함입니다.
// 동작 방식 (HOW): 64크기의 워크그룹 내 스레드들이 1D 인덱스를 계산하고, 범위를 초과하지 않으면 수학 함수 exp를 이용해 시그모이드 수식을 계산 후 y 배열에 씁니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 역할: 유니폼 버퍼에서 배열의 총 원소 개수를 읽어옵니다.
  let num_elements = params.size;
  
  // 변수: workgroups_x
  // 역할: 2차원 워크그룹 인덱스를 1차원 인덱스로 변환하기 위해 X축 워크그룹 수를 읽어옵니다.
  let workgroups_x = params.workgroups_x;
  
  // 변수: idx
  // 역할: 현재 쓰레드가 처리해야 할 1차원 데이터 인덱스
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 조건문: 배열 경계 확인
  // 역할: 인덱스가 실제 배열의 범위를 벗어날 경우 셰이더 실행을 조기 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 변수 y 갱신
  // 역할: x 배열의 값에 시그모이드 공식을 적용한 결과를 y 배열에 저장합니다.
  y[idx] = 1.0 / (1.0 + exp(-x[idx]));
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const TANH_WGSL = `
// 구조체: Params
// 목적: Tanh 커널 연산 시 필요한 설정값을 제공합니다.
// 작동 방식: 배열의 전체 요소 수와 워크그룹의 X축 크기를 포함합니다.
struct Params {
  // 변수: size
  // 목적: 전체 연산 대상 원소의 개수를 저장합니다.
  // 작동 방식: 범위를 초과하는 메모리 접근을 방지하는 기준으로 쓰입니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 워크그룹 수를 저장합니다.
  // 작동 방식: 3차원 스레드 ID를 1차원 인덱스로 변환할 때 곱해집니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 유니폼 구조체의 메모리 오프셋 규칙을 준수합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 유니폼 구조체의 메모리 오프셋 규칙을 준수합니다.
  pad3: u32,
};

// 변수: params
// 목적: 커널의 설정값을 가지고 있는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0을 통해 GPU에 전달되어 읽기 전용으로 참조됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: x
// 목적: Tanh 함수를 적용할 입력 텐서 데이터를 담고 있는 버퍼입니다.
// 작동 방식: 바인딩 1에 매핑되며 원본 데이터를 제공합니다.
@group(0) @binding(1) var<storage, read> x: array<f32>;

// 변수: y
// 목적: Tanh 함수의 계산 결과를 저장할 출력 버퍼입니다.
// 작동 방식: 바인딩 2에 매핑되며 계산된 활성화 값이 각 인덱스에 저장됩니다.
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

// 함수: main
// 목적: 병렬 스레드를 이용하여 배열의 각 요소에 대해 Tanh(쌍곡탄젠트) 함수를 계산합니다.
// 작동 방식: 내장 함수인 tanh()를 호출하여 y 배열에 저장합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 배열의 전체 요소 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 size를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기입니다.
  // 작동 방식: 유니폼 버퍼에서 workgroups_x를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 스레드가 처리할 1차원 배열의 인덱스입니다.
  // 작동 방식: global_id 정보를 바탕으로 평면화된 인덱스를 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 인덱스 범위를 초과한 스레드가 실행되는 것을 막습니다.
  // 작동 방식: idx가 num_elements 이상일 경우 함수를 조기 종료(return)합니다.
  if (idx >= num_elements) {
    return;
  }

  // 연산: y[idx] 기록
  // 목적: 특정 요소의 Tanh 값을 계산하여 저장합니다.
  // 작동 방식: WGSL 내장 함수 tanh(x[idx])를 호출하고 그 결과를 y[idx]에 씁니다.
  y[idx] = tanh(x[idx]);
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const SIGMOID_BACKWARD_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터들을 정의합니다. 메모리 정렬을 위해 패딩 변수가 포함되어 있습니다.
// 작동 방식: size와 workgroups_x 정보를 포함하여 작업 스레드가 자신의 위치를 파악할 수 있게 합니다.
struct Params {
  // 변수: size
  // 목적: 처리해야 할 전체 요소의 총 개수를 저장합니다.
  // 작동 방식: 배열의 범위를 초과하는 접근을 방지하는 기준값으로 사용됩니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수를 저장합니다.
  // 작동 방식: 2차원 워크그룹 인덱스를 1차원 전역 인덱스로 변환할 때 곱해지는 계수로 사용됩니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 접근 성능을 최적화하고 데이터 구조의 규격을 맞추는 역할을 합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 접근 성능을 최적화하고 데이터 구조의 규격을 맞추는 역할을 합니다.
  pad3: u32,
};

// 변수: params
// 목적: 외부에서 전달되는 설정값들을 저장하는 유니폼 버퍼(Uniform buffer) 변수입니다.
// 작동 방식: 바인딩 그룹 0, 바인딩 0에 매핑되어 워크그룹 실행 시 필요한 메타데이터를 제공합니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: grad
// 목적: 역전파(Backpropagation) 단계에서 이전 층(layer)으로부터 전달받은 손실(loss)의 기울기(gradient)를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 1에 매핑되며, 최종 기울기를 계산할 때 곱해지는 입력값으로 쓰입니다.
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// 변수: sigmoid_output
// 목적: 순전파(Forward propagation) 단계에서 미리 계산되었던 Sigmoid 함수의 출력 결과를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 2에 매핑되며, Sigmoid 미분 공식을 적용하기 위한 상태값으로 사용됩니다.
@group(0) @binding(2) var<storage, read> sigmoid_output: array<f32>;

// 변수: output
// 목적: 계산된 Sigmoid 함수의 역전파 기울기 결과를 저장할 읽기/쓰기 가능 버퍼입니다.
// 작동 방식: 바인딩 0, 바인딩 3에 매핑되며, 각 스레드에서 계산된 최종 미분값이 이곳에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: Sigmoid 함수의 역전파(Backward) 연산을 병렬로 수행하는 메인 컴퓨트 셰이더(Compute Shader) 진입점입니다.
// 작동 방식: Sigmoid 미분 공식인 'sigmoid_output * (1 - sigmoid_output)'을 사용하여 이전 기울기 'grad'와 곱한 뒤 최종 기울기를 계산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산해야 할 총 원소의 개수를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 size 필드를 읽어와 저장합니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 workgroups_x 필드를 읽어와 저장합니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 실행 중인 스레드가 담당할 1차원 데이터 인덱스를 계산합니다.
  // 작동 방식: 3차원인 global_id 값을 바탕으로, Y축 인덱스에 (X축 워크그룹 개수 * 워크그룹 크기 64)를 곱하고 X축 인덱스를 더해 평면화(flatten)된 인덱스를 구합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 범위를 벗어난 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: 계산된 인덱스(idx)가 처리해야 할 전체 요소 수(num_elements) 이상인지 확인합니다.
  if (idx >= num_elements) {
    // 유효 범위를 초과하면 아무 연산도 수행하지 않고 함수를 종료합니다.
    return;
  }

  // 연산: output[idx] 갱신
  // 목적: 최종적으로 입력 노드에 전달할 기울기(Gradient) 값을 도출하여 저장합니다.
  // 작동 방식: 체인 룰(Chain rule)에 의해 '상류에서 온 기울기(grad[idx])' * '로컬 미분값(sigmoid_output[idx] * (1.0 - sigmoid_output[idx]))'을 연산한 후 배열에 기록합니다.
  output[idx] = grad[idx] * sigmoid_output[idx] * (1.0 - sigmoid_output[idx]);
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const TANH_BACKWARD_WGSL = `
// 구조체: Params
// 목적: WGSL 커널에서 사용할 유니폼 파라미터들을 정의합니다. 메모리 정렬을 위해 패딩 변수가 포함되어 있습니다.
// 작동 방식: 배열 크기(size)와 2차원 워크그룹의 X축 크기(workgroups_x)를 제공합니다.
struct Params {
  // 변수: size
  // 목적: 연산할 전체 배열 요소의 개수입니다.
  // 작동 방식: 배열 범위를 초과하는 인덱스 접근을 차단하기 위한 경계값으로 쓰입니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향 워크그룹의 개수입니다.
  // 작동 방식: 3D 워크그룹 인덱스를 1D 전역 인덱스로 변환할 때 필요합니다.
  workgroups_x: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬을 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트의 배수로 맞춰 GPU 메모리 접근 오류를 방지합니다.
  pad3: u32,
};

// 변수: params
// 목적: 커널의 설정값을 가지고 있는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0을 통해 GPU에 전달되어 읽기 전용으로 참조됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: grad
// 목적: 역전파 시 이전 층(상류)으로부터 전달받은 손실 기울기를 저장하는 버퍼입니다.
// 작동 방식: 바인딩 1에 매핑되어 최종 미분값 계산에 곱해지는 입력값으로 쓰입니다.
@group(0) @binding(1) var<storage, read> grad: array<f32>;

// 변수: tanh_output
// 목적: 순전파 단계에서 이미 계산되었던 Tanh 함수의 출력 결과를 저장하는 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 2에 매핑되며, Tanh 미분 공식을 적용하기 위한 상태값으로 사용됩니다.
@group(0) @binding(2) var<storage, read> tanh_output: array<f32>;

// 변수: output
// 목적: 계산된 Tanh 함수의 역전파 기울기 결과를 저장할 읽기/쓰기 가능 버퍼입니다.
// 작동 방식: 바인딩 3에 매핑되며, 각 스레드에서 계산된 최종 미분값이 이곳에 기록됩니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: Tanh 함수의 역전파(Backward) 연산을 병렬로 수행하는 메인 컴퓨트 셰이더 진입점입니다.
// 작동 방식: Tanh 미분 공식 '1 - tanh_output^2'에 이전 기울기 'grad'를 곱하여 최종 기울기를 계산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산해야 할 총 원소의 개수를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 size 필드를 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 워크그룹의 크기를 로컬 변수로 가져옵니다.
  // 작동 방식: params 구조체에서 workgroups_x 필드를 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 실행 중인 스레드가 담당할 1차원 데이터 인덱스를 계산합니다.
  // 작동 방식: 3차원인 global_id 값을 바탕으로, Y축 인덱스에 (X축 워크그룹 개수 * 64)를 곱하고 X축 인덱스를 더해 평면화합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 유효한 데이터 범위를 벗어난 스레드가 실행되는 것을 방지합니다.
  // 작동 방식: idx가 num_elements 이상인지 확인하여 맞으면 함수를 종료합니다.
  if (idx >= num_elements) {
    return;
  }

  // 연산: output[idx] 갱신
  // 목적: 최종적으로 입력 노드에 전달할 기울기(Gradient) 값을 도출하여 저장합니다.
  // 작동 방식: 체인 룰(Chain rule)에 의해 '상류에서 온 기울기(grad[idx])' * '로컬 미분값(1.0 - tanh_output[idx] * tanh_output[idx])'을 계산해 기록합니다.
  output[idx] = grad[idx] * (1.0 - tanh_output[idx] * tanh_output[idx]);
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     * 수정 이력:
     * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const FILL_WGSL = `
/**
 * 이 구조체(Params)는 텐서를 특정 값으로 채우기 위한 설정 정보를 담고 있습니다.
 * GPU에 유니폼 버퍼를 통해 전달되며, 16바이트 정렬을 맞추기 위해 패딩을 포함합니다.
 */
struct Params {
  numElements: u32, // 값을 채울 배열의 전체 요소 개수입니다.
  value: f32, // 배열을 채울 특정 단일 부동 소수점 값입니다.
  pad1: u32, // 메모리 정렬을 위해 추가된 패딩용 변수입니다.
  pad2: u32, // 메모리 정렬을 위해 추가된 두 번째 패딩용 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에서 읽어들일 유니폼 데이터입니다.
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // 채워진 값이 쓰여질 출력 버퍼입니다.

/**
 * main 함수는 출력 배열의 모든 요소에 지정된 값을 병렬로 기록합니다.
 * 텐서를 특정 상수값으로 초기화하는 fill 연산을 GPU에서 고속으로 수행하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let num_elements = params.numElements; // 전체 요소 개수를 유니폼 변수에서 가져옵니다.
  let idx = global_id.x; // 현재 스레드가 담당할 1D 배열 내의 인덱스입니다.
  
  // 계산된 인덱스가 전체 배열 크기보다 크거나 같다면 작업을 수행하지 않고 종료합니다.
  if (idx >= num_elements) {
    return;
  }
  
  // 지정된 인덱스 위치에 설정된 상수 값(params.value)을 저장합니다.
  output[idx] = params.value;
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const SUM_WGSL = `
// 구조체: Params
// 목적: 합계(Sum) 연산에 필요한 메타데이터와 패딩을 정의합니다.
// 작동 방식: 배열의 전체 크기를 제공하며, 16바이트 정렬을 준수합니다.
struct Params {
  // 변수: numElements
  // 목적: 더해야 할 입력 배열의 전체 원소 개수를 나타냅니다.
  // 작동 방식: 전역 인덱스가 유효 범위를 벗어나는지 검사하는 용도로 사용됩니다.
  numElements: u32,
  // 변수: pad1
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트 배수로 맞춰 메모리 접근 성능을 높입니다.
  pad1: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트 배수로 맞춰 메모리 접근 성능을 높입니다.
  pad2: u32,
  // 변수: pad3
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: 구조체 크기를 16바이트 배수로 맞춰 메모리 접근 성능을 높입니다.
  pad3: u32,
};

// 변수: params
// 목적: 유니폼 버퍼를 통해 워크그룹 외부에서 메타데이터를 주입받습니다.
// 작동 방식: 바인딩 0에 할당되어 전체 요소 개수를 모든 스레드에 제공합니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 합계를 구할 대상이 되는 데이터를 담은 읽기 전용 버퍼입니다.
// 작동 방식: 바인딩 1에 할당되며, 각 스레드가 자신의 위치에 해당하는 값을 읽어옵니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 목적: 각 워크그룹 내에서의 부분 합계(Partial sum)를 저장할 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되며, 최종적으로 워크그룹 개수만큼의 결과가 저장됩니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 변수: s_data
// 목적: 워크그룹 내 스레드들이 공유하는 로컬 메모리(Shared memory)입니다.
// 작동 방식: 256 크기의 배열로 할당되어 빠른 Reduction(축소) 연산을 위한 캐시 역할을 합니다.
var<workgroup> s_data: array<f32, 256>;

// 함수: main
// 목적: 배열 요소들의 총합을 구하기 위한 병렬 Reduction(축소) 알고리즘을 수행합니다.
// 작동 방식: 워크그룹 내에서 공유 메모리를 사용하여 트리(Tree) 구조로 단계별 덧셈을 수행합니다.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  // 변수: gid
  // 목적: 전체 스레드 중 현재 스레드의 고유 1차원 전역 인덱스입니다.
  // 작동 방식: global_id.x 값을 가져와 입력 데이터 접근에 사용합니다.
  let gid = global_id.x;

  // 변수: lid
  // 목적: 현재 워크그룹 내에서의 로컬 인덱스(0~255)입니다.
  // 작동 방식: local_id.x 값을 가져와 공유 메모리 접근 및 축소 연산 인덱스로 사용합니다.
  let lid = local_id.x;

  // 변수: wid
  // 목적: 현재 속해 있는 워크그룹의 고유 ID입니다.
  // 작동 방식: workgroup_id.x 값을 가져와 부분 합 결과를 저장할 위치를 결정합니다.
  let wid = workgroup_id.x;
  
  // 제어문: if-else
  // 목적: 입력 데이터를 로컬 공유 메모리에 복사하면서, 범위를 벗어난 공간을 0으로 초기화합니다.
  // 작동 방식: gid가 유효한 원소 범위 안에 있으면 input[gid]를, 벗어나면 0.0을 s_data에 할당합니다.
  if (gid < params.numElements) {
    s_data[lid] = input[gid];
  } else {
    s_data[lid] = 0.0;
  }
  
  // 동기화: workgroupBarrier()
  // 목적: 워크그룹 내의 모든 스레드가 공유 메모리에 데이터를 쓸 때까지 대기합니다.
  // 작동 방식: 초기 데이터 적재(Load)가 완전히 끝난 뒤에만 다음 Reduction 단계를 진행하도록 보장합니다.
  workgroupBarrier();
  
  // 반복문: for
  // 목적: 워크그룹 내 256개의 요소를 트리 기반 병렬 Reduction 방식으로 더합니다.
  // 작동 방식: s 변수를 128부터 시작하여 0보다 클 때까지 절반으로 줄여가며 (비트 시프트 연산) 부분 합을 구합니다.
  for (var s = 128u; s > 0u; s >>= 1u) {
    // 제어문: if
    // 목적: 유효한 절반의 스레드들만 덧셈 연산에 참여하도록 제한합니다.
    // 작동 방식: 로컬 인덱스가 현재 단계의 s보다 작을 경우에만 s_data[lid]와 s_data[lid + s]를 더합니다.
    if (lid < s) {
      s_data[lid] = s_data[lid] + s_data[lid + s];
    }
    // 동기화: workgroupBarrier()
    // 목적: 다음 단계의 Reduction으로 넘어가기 전, 현재 단계의 덧셈이 모든 스레드에서 완료되었는지 확인합니다.
    // 작동 방식: 모든 스레드가 동기화 지점에 도달할 때까지 실행을 일시 중단합니다.
    workgroupBarrier();
  }
  
  // 제어문: if
  // 목적: 워크그룹 내 최종 합산 결과(s_data[0])를 전역 출력 버퍼에 단 한 번만 기록합니다.
  // 작동 방식: 로컬 인덱스가 0번인 스레드만 대표로 output[wid]에 s_data[0] 값을 할당합니다.
  if (lid == 0u) {
    output[wid] = s_data[0];
  }
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:14:52 +0900 (commit c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     * 수정 이력:
     * - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const MAX_WGSL = `
/**
 * 이 구조체(Params)는 텐서의 최댓값을 구하는 Reduction(리덕션) 연산에 필요한 정보를 담고 있습니다.
 * 요소의 전체 개수를 전달하여 버퍼 경계를 넘는 접근을 방지하기 위해 존재합니다.
 */
struct Params {
  numElements: u32, // 최댓값을 찾을 전체 배열 원소의 개수입니다.
  pad1: u32, // WebGPU의 16바이트 정렬을 맞추기 위한 빈 패딩 변수입니다.
  pad2: u32, // 16바이트 정렬을 위한 두 번째 패딩 변수입니다.
  pad3: u32, // 16바이트 정렬을 위한 세 번째 패딩 변수입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 전달되는 메타데이터 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 최댓값을 탐색할 원본 입력 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 워크그룹별 부분 최댓값이 저장될 출력 텐서입니다.

// 하나의 워크그룹(256개의 스레드) 내에서 데이터를 공유하고 리덕션을 수행하기 위해 존재하는 공유 메모리 공간입니다.
var<workgroup> s_data: array<f32, 256>;

/**
 * main 함수는 트리 기반의 리덕션(Tree-based Reduction) 알고리즘을 사용하여 배열 내 원소들의 최댓값을 계산합니다.
 * 방대한 데이터를 병렬로 빠르게 비교압축하기 위해 공유 메모리(s_data)와 배리어(barrier) 동기화를 사용합니다.
 */
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
  let gid = global_id.x; // 글로벌 단위에서 현재 스레드의 1차원 인덱스입니다.
  let lid = local_id.x; // 워크그룹 내부에서 현재 스레드의 1차원 인덱스(0~255)입니다.
  let wid = workgroup_id.x; // 현재 스레드가 속한 워크그룹의 ID(인덱스)입니다.
  
  // 글로벌 인덱스가 데이터 크기 이내라면 입력 데이터를, 벗어난다면 부동소수점의 최소값(-FLT_MAX)을 공유 메모리에 로드합니다.
  if (gid < params.numElements) {
    s_data[lid] = input[gid];
  } else {
    s_data[lid] = -3.402823e+38; // 쓰레기값을 방지하기 위한 최소값 초기화입니다.
  }
  
  // 공유 메모리 로드가 완전히 끝날 때까지 워크그룹 내의 모든 스레드를 대기시킵니다.
  workgroupBarrier();
  
  // 트리 기반 병렬 리덕션 루프입니다.
  // 활성화된 스레드 수를 절반씩 줄여가면서(128 -> 64 -> ... -> 1) 두 요소씩 비교해 최댓값을 찾습니다.
  for (var s = 128u; s > 0u; s >>= 1u) {
    // 현재 단계에서 값을 비교하고 갱신할 권한이 있는 스레드만 실행합니다.
    if (lid < s) {
      s_data[lid] = max(s_data[lid], s_data[lid + s]); // 자신의 값과 s만큼 떨어진 옆의 값을 비교해 큰 값을 저장합니다.
    }
    // 데이터 경합(Data Race)을 막고 다음 단계를 안전하게 수행하기 위해 스레드 동기화를 수행합니다.
    workgroupBarrier();
  }
  
  // 리덕션이 완료되면 공유 메모리의 0번 인덱스에 현재 워크그룹의 전체 최댓값이 남게 됩니다.
  // 0번 스레드가 이를 대표로 전역 출력 버퍼에 기록합니다.
  if (lid == 0u) {
    output[wid] = s_data[0];
  }
}
`;

    /**
     * 파일 생성: 2026-08-12 12:14:52
     * 수정 내역:
     * - 2026-08-12 12:14:52: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories (c2ee1bbf60255f375f779eba2ff8b1270c48b6e6)
     */
    const SUM_AXIS_WGSL = `
// 구조체: Params
// 목적: 특정 축(Axis)을 기준으로 합계를 구할 때 필요한 차원 정보를 제공합니다.
// 작동 방식: 행(M)과 열(N)의 크기를 정의하고, 16바이트 메모리 정렬을 맞춥니다.
struct Params {
  // 변수: M
  // 목적: 누적(sum) 연산이 수행될 행(Row)의 크기(축의 길이)를 지정합니다.
  // 작동 방식: 반복문에서 행의 인덱스가 M에 도달할 때까지 합계를 구합니다.
  M: u32,
  // 변수: N
  // 목적: 스레드들이 병렬로 처리할 열(Column)의 개수입니다.
  // 작동 방식: 각 스레드가 N개의 열 중 하나를 담당하여 각 열의 합계를 독립적으로 계산합니다.
  N: u32,
  // 변수: pad1
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 읽기 성능 저하를 방지하기 위해 빈 공간을 둡니다.
  pad1: u32,
  // 변수: pad2
  // 목적: 16바이트 메모리 정렬(Alignment)을 맞추기 위한 패딩입니다.
  // 작동 방식: GPU 메모리 읽기 성능 저하를 방지하기 위해 빈 공간을 둡니다.
  pad2: u32,
};

// 변수: params
// 목적: 커널 실행 시 M, N 등의 차원 정보를 담아 전달하는 유니폼 버퍼입니다.
// 작동 방식: 바인딩 0에 매핑되어 워크그룹 내에서 공유됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 목적: 합계를 구할 2차원(혹은 1차원으로 평면화된) 배열 데이터를 저장하는 버퍼입니다.
// 작동 방식: 바인딩 1에 할당되며 읽기 전용으로 접근합니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 목적: 특정 축을 기준으로 축소(Reduce)된 결과를 저장할 버퍼입니다.
// 작동 방식: 바인딩 2에 할당되며, N 크기의 배열로 결과가 쓰여집니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 목적: 주어진 데이터의 첫 번째 축(Row)을 기준으로 열 단위 합계를 병렬 연산합니다.
// 작동 방식: 각 스레드가 하나의 열(col)을 담당하고 모든 행(row)을 순회하며 합산합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: col
  // 목적: 현재 스레드가 담당할 열(Column)의 인덱스입니다.
  // 작동 방식: global_id.x 값을 이용하여 계산할 열 위치를 특정합니다.
  let col = global_id.x;
  
  // 제어문: if
  // 목적: 할당된 스레드의 인덱스가 실제 열 개수(N)를 초과하는지 검사합니다.
  // 작동 방식: col이 N 이상이면 유효하지 않은 스레드이므로 즉시 종료(return)합니다.
  if (col >= params.N) {
    return;
  }
  
  // 변수: sum
  // 목적: 특정 열에 대한 총합을 누적할 로컬 변수입니다.
  // 작동 방식: 0.0으로 초기화된 후 루프를 돌면서 요소의 값을 계속 더해나갑니다.
  var sum = 0.0;
  
  // 반복문: for
  // 목적: 행(Row) 방향으로 데이터를 탐색하며 각 요소를 더합니다.
  // 작동 방식: row를 0부터 M-1까지 1씩 증가시키며 \`sum += input[row * N + col]\` 연산을 수행합니다.
  for (var row = 0u; row < params.M; row = row + 1u) {
    sum += input[row * params.N + col];
  }
  
  // 연산: output 배열 기록
  // 목적: 반복문이 끝난 후 계산된 최종 열 합계를 결과 버퍼에 저장합니다.
  // 작동 방식: 해당 스레드가 담당한 열 인덱스(col) 위치에 누적된 sum을 기록합니다.
  output[col] = sum;
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:14:52 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     */
    const AXPY_WGSL = `
/**
 * @struct Params
 * @brief AXPY (Y = Y - alpha * X 형태, 여기서는 Parameter Update) 연산의 파라미터를 담고 있습니다. (What)
 * 머신러닝의 경사하강법(Gradient Descent) 시 가중치를 학습률(learning rate)에 비례하여 업데이트하기 위해 존재합니다. (Why)
 */
struct Params {
  // 업데이트를 수행할 전체 요소(가중치/파라미터)의 개수입니다.
  numElements: u32,
  // 학습률(learning rate, lr)입니다. 그레이디언트(grad)가 파라미터에 미치는 영향을 조절합니다.
  lr: f32,
  // 16바이트(float4) 정렬을 맞추기 위해 사용되는 패딩입니다.
  pad1: u32,
  // 16바이트(float4) 정렬을 맞추기 위해 사용되는 패딩입니다.
  pad2: u32,
};

// params: 균일한 크기를 가지는 파라미터 버퍼입니다. 업데이트에 필요한 총 요소 수와 학습률 등을 포함합니다.
@group(0) @binding(0) var<uniform> params: Params;
// grad: 파라미터에 대한 기울기(Gradient) 값을 담고 있는 배열입니다. (읽기 전용)
@group(0) @binding(1) var<storage, read> grad: array<f32>;
// param: 현재 모델의 가중치(파라미터) 배열입니다. 읽고 쓰기가 가능하며 이 배열 자체에 결과를 덮어씌웁니다(In-place update).
@group(0) @binding(2) var<storage, read_write> param: array<f32>;

/**
 * @function main
 * @brief 그레이디언트 값에 학습률을 곱한 뒤 기존 파라미터에서 빼는 방식(param = param - lr * grad)으로 값을 갱신합니다. (What)
 * GPU 코어들을 병렬로 사용하여 수많은 모델 파라미터를 한 번에 업데이트하기 위해(Why) 실행되는 메인 셰이더입니다.
 * @param global_id 워크그룹 내 스레드의 3차원 전역 인덱스 변수입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 1차원 전역 인덱스를 가져옵니다. 각 스레드가 하나의 파라미터를 담당합니다. (How)
  let idx = global_id.x;
  
  // 현재 인덱스가 처리해야 할 요소 개수(numElements)를 넘어갔는지 검사합니다. (What)
  // 배열 크기 이상의 메모리 접근을 방지하기 위함입니다. (Why)
  if (idx >= params.numElements) {
    return;
  }
  
  let g = grad[idx];
  // Numerical Safety: Guard against NaN/Inf gradient poisoning (IEEE 754 in WGSL)
  let safe_g = select(clamp(g, -1e6, 1e6), 0.0, g != g);
  param[idx] = param[idx] - params.lr * safe_g;
}
`;

    /**
     * 생성일: 2026-08-12T12:23:09+09:00
     * 수정 이력:
     * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
     */
    const PAD_WGSL = `
// 구조체: Params
// 역할 (WHAT): 텐서 패딩 연산에 필요한 모든 형태(Shape), 보폭(Stride), 설정 변수들을 담고 있는 구조체입니다.
// 목적 (WHY): 패딩 된 새로운 텐서를 생성하기 위해 원본 텐서의 좌표와 출력 텐서의 좌표를 매핑하는 데 필요한 정보를 제공하기 위함입니다.
// 동작 방식 (HOW): 각 차원에 대한 크기, 원본/출력 메모리 보폭 정보, 추가할 패딩 값 등을 참조하여 변환된 인덱스를 계산합니다.
struct Params {
  // 변수: num_elements
  // 역할: 패딩이 완료된 최종 출력 텐서의 전체 요소 개수
  num_elements: u32,
  // 변수: rank
  // 역할: 텐서의 차원(Rank) 수
  rank: u32,
  // 변수: pad_val
  // 역할: 빈 공간에 채워 넣을 상수 값(패딩 값)
  pad_val: f32,
  // 변수: _pad
  // 역할: WebGPU 메모리 정렬(16바이트)을 맞추기 위한 여분(padding) 변수
  _pad: u32,
  // 변수: in_strides
  // 역할: 최대 8차원까지 지원하는 원본 입력 텐서의 차원별 메모리 보폭(Stride) 배열
  in_strides: array<u32, 8>,
  // 변수: out_strides
  // 역할: 패딩 적용 후 출력 텐서의 차원별 메모리 보폭 배열
  out_strides: array<u32, 8>,
  // 변수: pad_before
  // 역할: 각 차원의 앞부분(before)에 추가되는 패딩의 크기를 저장하는 배열
  pad_before: array<u32, 8>,
  // 변수: in_shape
  // 역할: 입력 텐서의 원래 차원별 크기(Shape)를 저장하는 배열
  in_shape: array<u32, 8>,
};

// 변수: params
// 역할: 패딩 연산의 메타데이터를 저장하는 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 역할: 패딩 되기 전의 원본 데이터가 저장되어 있는 스토리지 버퍼
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 역할: 패딩 된 결과 데이터가 기록될 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 출력 텐서의 각 인덱스를 기준으로 원본 인덱스를 역추적하여 값을 복사하거나 패딩 값을 채웁니다.
// 목적 (WHY): 입력 배열 주변에 원하는 크기와 값으로 여백(패딩)을 추가하여 크기가 확장된 텐서를 반환하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 출력 1D 인덱스(idx)를 받아 다차원 좌표(coord)로 변환한 후, 이 좌표가 원본 텐서 영역 내인지 확인하여 원본 값을 쓰거나 pad_val을 채웁니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: idx
  // 역할: 현재 스레드가 담당하는 출력 텐서의 1차원 전역 인덱스
  let idx = global_id.x;
  
  // 조건문: 인덱스 범위 확인
  // 역할: idx가 결과 텐서의 전체 크기를 넘어서면 실행을 중지하여 유효하지 않은 메모리 접근을 방지합니다.
  if (idx >= params.num_elements) { return; }

  // 변수: temp
  // 역할: 1차원 인덱스를 다차원 좌표로 분해할 때 남은 인덱스 값을 저장 및 갱신하기 위한 임시 변수
  var temp = idx;
  // 변수: in_idx
  // 역할: 역계산된 원본 텐서의 1차원 인덱스를 누적할 변수
  var in_idx = 0u;
  // 변수: in_bounds
  // 역할: 현재 계산하는 출력 좌표가 원본 텐서의 범위 안에 속해 있는지를 나타내는 불리언 플래그
  var in_bounds = true;

  // 반복문: for 루프 (차원 탐색)
  // 역할 (WHAT): 최고 차원부터 최하 차원까지 각 차원의 좌표를 구하고, 이를 이용해 원본 입력 텐서의 플랫(flat) 인덱스를 누적 연산합니다.
  // 목적 (WHY): N차원(최대 8차원) 데이터를 1차원 배열로 평탄화(Flatten)한 메모리 구조에서 정확한 매핑을 계산하기 위함입니다.
  // 동작 방식 (HOW): 나누기와 나머지 연산을 사용해 현재 차원의 좌표(coord)를 구한 뒤, 원본 텐서 구간(pad_before ~ pad_before + in_shape)에 속하는지 검사합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: coord
    // 역할: 출력 텐서의 i번째 차원에 대한 구체적 좌표(인덱스)
    let coord = temp / params.out_strides[i];
    
    // 변수: temp 갱신
    // 역할: 다음 하위 차원 계산을 위해 남은 나머지 값을 임시 변수에 대입합니다.
    temp = temp % params.out_strides[i];
    
    // 조건문: 원본 영역 이탈 확인
    // 역할: 계산된 해당 차원의 좌표가 패딩 영역(원본 데이터가 없는 곳)인지 판단합니다.
    if (coord < params.pad_before[i] || coord >= params.pad_before[i] + params.in_shape[i]) {
      // 변수: in_bounds 갱신
      // 역할: 영역 바깥이므로 in_bounds를 false로 변경하고 루프를 탈출합니다.
      in_bounds = false;
      break;
    }
    
    // 변수: in_coord
    // 역할: 출력 텐서 좌표에서 앞부분 패딩(pad_before)을 빼서 원본 텐서 기준의 순수 좌표를 구합니다.
    let in_coord = coord - params.pad_before[i];
    
    // 변수: in_idx 누적
    // 역할: 구한 원본 좌표에 해당 차원의 보폭(in_strides)을 곱하여 1D 원본 인덱스를 점진적으로 계산합니다.
    in_idx = in_idx + in_coord * params.in_strides[i];
  }

  // 조건문: 값 삽입 결정
  // 역할: 구해진 플래그(in_bounds)를 바탕으로 배열에 원본 데이터를 쓸지, 패딩 값을 쓸지 분기합니다.
  if (in_bounds) {
    // 변수 output 갱신 (원본)
    // 역할: 출력 배열에 입력 배열의 데이터를 그대로 복사합니다.
    output[idx] = input[in_idx];
  } else {
    // 변수 output 갱신 (패딩)
    // 역할: 출력 배열에 미리 설정해 둔 패딩 상수 값(pad_val)을 삽입합니다.
    output[idx] = params.pad_val;
  }
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:23:09 +0900 (commit fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
     * 수정 이력:
     * - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
     */
    const GATHER_WGSL = `
/**
 * 이 구조체(Params)는 gather 연산에 필요한 형태(shape), 차원(stride), 대상 차원(dim) 정보를 담고 있습니다.
 * 다차원 텐서 인덱싱을 1차원 메모리에서 올바르게 계산하기 위한 정보를 제공하기 위해 존재합니다.
 */
struct Params {
  num_elements: u32, // 출력 텐서의 총 원소 개수입니다.
  dim: u32, // 요소를 수집할(gather) 대상 차원(axis)의 인덱스입니다.
  rank: u32, // 텐서의 차원 수 (랭크)입니다.
  _pad: u32, // 16바이트 메모리 정렬을 위한 패딩입니다.
  x_strides: array<u32, 8>, // 원본 입력 텐서의 각 차원별 스트라이드(보폭)입니다.
  out_strides: array<u32, 8>, // 출력 텐서의 각 차원별 스트라이드(보폭)입니다.
  x_shape: array<u32, 8>, // 원본 입력 텐서의 모양(각 차원의 크기)입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // 메타데이터 및 형태 정보가 담긴 유니폼 데이터입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // 수집 대상이 되는 원본 데이터 배열입니다.
@group(0) @binding(2) var<storage, read> index: array<f32>; // 수집할 인덱스를 지정하는 배열입니다.
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // 수집된 데이터가 쓰여질 결과 배열입니다.

/**
 * main 함수는 다차원 텐서에서 지정된 축(dim)을 기준으로 index 배열에 명시된 위치의 값들을 가져와 출력 텐서를 생성합니다.
 * PyTorch/NumPy의 gather 연산을 GPU에서 병렬 처리하기 위해 존재하며, 각 스레드는 출력 배열의 한 요소에 대응합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x; // 출력 텐서에서 현재 스레드가 처리할 1D 위치(인덱스)입니다.
  
  // 현재 처리할 인덱스가 전체 요소 수를 초과하면 실행을 중단합니다.
  if (idx >= params.num_elements) { return; }

  var temp = idx; // 다차원 좌표를 계산하기 위해 인덱스를 임시 변수에 복사합니다.
  var in_idx = 0u; // 입력 텐서에서 실제 참조해야 할 1D 메모리 인덱스를 누적할 변수입니다.

  // 출력 텐서의 각 차원(0부터 rank-1까지)에 대해 루프를 돕니다.
  // 이 루프는 출력 텐서의 1D 인덱스(idx)를 다차원 좌표로 변환하고, 이를 다시 입력 텐서의 1D 인덱스(in_idx)로 매핑합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    let coord = temp / params.out_strides[i]; // 현재 차원 i에서의 다차원 좌표 값입니다.
    temp = temp % params.out_strides[i]; // 다음 하위 차원 좌표 계산을 위해 나머지를 구합니다.
    
    // 현재 차원이 수집 대상 차원(dim)인 경우, 계산된 좌표 대신 index 배열에서 값을 읽어옵니다.
    if (i == params.dim) {
      let idx_val = u32(index[idx]); // index 배열에서 대상 인덱스를 가져와 정수로 변환합니다.
      in_idx = in_idx + idx_val * params.x_strides[i]; // 가져온 인덱스에 원본 텐서의 해당 차원 스트라이드를 곱해 누적합니다.
    } else {
      // 수집 대상 차원이 아닌 경우, 출력 텐서와 동일한 좌표를 유지합니다.
      in_idx = in_idx + coord * params.x_strides[i]; // 동일한 좌표에 원본 텐서의 해당 차원 스트라이드를 곱해 누적합니다.
    }
  }

  // 최종적으로 계산된 입력 텐서 인덱스(in_idx)의 값을 읽어 출력 텐서의 현재 인덱스(idx)에 저장합니다.
  output[idx] = input[in_idx];
}
`;

    /**
     * 생성일: 2026-08-12T12:23:09+09:00
     * 수정 이력:
     * - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
     */
    const SCATTER_WGSL = `
// 구조체: Params
// 역할 (WHAT): 스캐터(Scatter) 연산에 필요한 차원, 보폭 및 타겟 축 정보를 저장하는 구조체입니다.
// 목적 (WHY): 입력 데이터 텐서와 인덱스 텐서를 조합하여 출력 텐서의 어느 위치에 값을 기록할지 결정하기 위함입니다.
// 동작 방식 (HOW): 각 차원에 대한 크기(rank), 흩뿌릴 차원(dim), 인덱스/입력의 보폭 정보를 읽어와 좌표를 계산합니다.
struct Params {
  // 변수: num_elements
  // 역할: 처리할 입력 요소들의 전체 개수
  num_elements: u32,
  // 변수: dim
  // 역할: 인덱스 값으로 대체되어 흩뿌려질 대상 차원 축
  dim: u32,
  // 변수: rank
  // 역할: 텐서가 갖는 전체 차원 수
  rank: u32,
  // 변수: _pad
  // 역할: 메모 정렬을 위한 16바이트 패딩 변수
  _pad: u32,
  // 변수: x_strides
  // 역할: 출력 배열(입력과 동일한 형상을 가지는 베이스)의 각 차원별 메모리 보폭 배열
  x_strides: array<u32, 8>,
  // 변수: idx_strides
  // 역할: 인덱스 텐서의 각 차원별 메모리 보폭 배열
  idx_strides: array<u32, 8>,
};

// 변수: params
// 역할: 스캐터 연산의 메타데이터를 담은 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: index
// 역할: 흩뿌릴 위치 정보를 가지고 있는 인덱스 배열(읽기 전용 스토리지 버퍼)
@group(0) @binding(1) var<storage, read> index: array<f32>;

// 변수: src
// 역할: 출력 배열에 복사할 원본 값을 가지고 있는 소스 배열(읽기 전용 스토리지 버퍼)
@group(0) @binding(2) var<storage, read> src: array<f32>;

// 변수: output
// 역할: 원본 값들이 인덱스 배열의 지시에 따라 흩뿌려진 최종 결과물이 저장될 스토리지 버퍼
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 주어진 인덱스 텐서의 값에 따라 소스 데이터를 출력 텐서의 특정 위치에 저장합니다.
// 목적 (WHY): 특정 차원의 값을 인덱스로 치환하여(Scatter-Elements) 텐서 내 원하는 위치에 데이터를 쓰기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 1차원 ID를 다차원 좌표로 변환하고, 지정된 축(dim)에 대해서만 원래 좌표 대신 인덱스 텐서의 값을 좌표로 사용하여 출력 위치를 정합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: idx
  // 역할: 소스 데이터와 인덱스 데이터의 1차원 메모리 인덱스
  let idx = global_id.x;
  
  // 조건문: 데이터 경계 검사
  // 역할: 할당된 스레드의 인덱스가 전체 크기(num_elements)를 초과하는지 검사합니다.
  if (idx >= params.num_elements) { return; }

  // 변수: temp
  // 역할: 다차원 좌표로 분리해 나가기 위해 남은 인덱스 수치를 보관하는 임시 변수
  var temp = idx;
  // 변수: out_idx
  // 역할: 최종적으로 계산된 출력 배열의 1차원 메모리 인덱스를 누적할 변수
  var out_idx = 0u;

  // 반복문: for 루프 (모든 차원 순회)
  // 역할 (WHAT): 최상위 차원부터 0번째 차원까지 각 차원의 좌표를 구하고, 이를 이용해 출력 인덱스를 계산합니다.
  // 목적 (WHY): 1차원 인덱스를 다시 N차원 좌표로 풀고, 특정 차원(dim)에 대해서만 값을 교체하기 위해 필요합니다.
  // 동작 방식 (HOW): i가 dim과 같을 경우, 계산된 논리적 좌표 대신 index 배열에 있는 값을 가져와서 보폭을 곱하고, 그 외의 경우 원래 좌표에 보폭을 곱합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: coord
    // 역할: 현재 차원(i)에 해당하는 인덱스 텐서 기준의 다차원 논리 좌표
    let coord = temp / params.idx_strides[i];
    
    // 변수: temp 갱신
    // 역할: 다음 차원 계산을 위해 나머지 값을 임시 변수에 업데이트합니다.
    temp = temp % params.idx_strides[i];
    
    // 조건문: 타겟 차원(dim) 여부 검사
    // 역할: 현재 처리 중인 차원이 인덱스 값으로 대체할 타겟 차원인지 판단합니다.
    if (i == params.dim) {
      // 변수: idx_val
      // 역할: 인덱스 텐서(index)에서 해당 위치의 값을 정수로 캐스팅하여 얻은 치환될 좌표값
      let idx_val = u32(index[idx]);
      
      // 변수: out_idx 누적 (치환된 축)
      // 역할: 원래 좌표 대신 치환된 좌표(idx_val)에 출력 보폭(x_strides)을 곱해 더합니다.
      out_idx = out_idx + idx_val * params.x_strides[i];
    } else {
      // 변수: out_idx 누적 (일반 축)
      // 역할: 원래 논리적 좌표(coord)에 출력 보폭(x_strides)을 곱해 더합니다.
      out_idx = out_idx + coord * params.x_strides[i];
    }
  }

  // 주석: 엄밀한 원자성(Atomic)은 제공하지 않지만 인덱스가 겹치지 않는 단순 스캐터의 경우 정상 동작함.
  // 변수 output 갱신
  // 역할: 치환이 완료되어 도출된 출력 인덱스 위치(out_idx)에 소스 배열의 데이터(src[idx])를 저장합니다.
  // Not strictly atomic, but for simple scatter where indices are unique it's fine.
  output[out_idx] = src[idx];
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:23:09 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
     */
    const CAT_WGSL = `
/**
 * @struct Params
 * @brief 두 텐서를 특정 차원(dimension)을 기준으로 결합(concatenate)할 때 사용하는 파라미터 구조체입니다. (What)
 * 텐서의 형태와 크기를 기반으로 각 텐서에서 어떤 위치의 값을 가져올지 인덱스를 계산하기 위해 존재합니다. (Why)
 */
struct Params {
  // 결합이 완료된 결과 텐서의 전체 요소(element) 개수입니다.
  size: u32,
  // X축 워크그룹 수. 2차원 그리드 인덱싱을 1차원 인덱스로 풀기 위한 변수입니다.
  workgroups_x: u32,
  // 결합하려는 축(axis)에서 첫 번째 텐서(A)가 차지하는 차원의 크기입니다.
  a_dim: u32,
  // 결합하려는 축(axis)에서 두 번째 텐서(B)가 차지하는 차원의 크기입니다.
  b_dim: u32,
  // 결합 축(axis)보다 하위에 있는 차원들의 요소 개수 곱입니다(Stride). 
  // 상위 차원이나 배치(batch)를 뛰어넘기 위한 보폭 역할을 합니다. (How)
  stride: u32,
  // 메모리 정렬(16바이트)을 위한 패딩 변수 1입니다.
  pad1: u32,
  // 메모리 정렬을 위한 패딩 변수 2입니다.
  pad2: u32,
  // 메모리 정렬을 위한 패딩 변수 3입니다.
  pad3: u32,
};

// params: 결합 연산에 필요한 차원 및 크기 정보를 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 결합될 첫 번째 입력 텐서 데이터 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 결합될 두 번째 입력 텐서 데이터 배열입니다 (읽기 전용).
@group(0) @binding(2) var<storage, read> b: array<f32>;
// out: A와 B가 이어진(Concatenated) 결과가 저장되는 배열입니다.
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

/**
 * @function main
 * @brief 결과 텐서의 각 요소가 입력 텐서 A 혹은 B 중 어디서 와야 하는지를 계산하고 복사합니다. (What)
 * 병렬 인덱싱을 통하여 다차원 텐서의 결합 연산을 빠르게 수행하기 위해 만들어졌습니다. (Why)
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스 변수입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 파라미터에서 전체 데이터 개수를 가져옵니다.
  let num_elements = params.size;
  // 파라미터에서 X 방향 워크그룹 개수를 가져옵니다.
  let workgroups_x = params.workgroups_x;
  // 3차원 워크그룹 및 스레드 ID를 1차원 선형 인덱스로 변환합니다. (How)
  let idx = global_id.x + global_id.y * workgroups_x * 64u;
  
  // 계산된 인덱스가 전체 요소 개수를 넘어갈 경우 안전하게 함수를 종료합니다. (What)
  // 배열 범위를 벗어난 메모리에 대한 불법적인 쓰기를 방지하기 위해서입니다. (Why)
  if (idx >= num_elements) {
    return;
  }
  
  // 파라미터 구조체에서 내부 차원의 크기(stride)를 로드합니다.
  let stride = params.stride;
  // 파라미터 구조체에서 A 텐서의 결합 축 크기를 로드합니다.
  let a_dim = params.a_dim;
  // 파라미터 구조체에서 B 텐서의 결합 축 크기를 로드합니다.
  let b_dim = params.b_dim;
  
  // 결합된 이후 결과 텐서의 해당 축 길이를 계산합니다. (What)
  let out_dim_size = a_dim + b_dim;
  // 한 블록(결합 축 1개 단위 + 하위 차원 전체)이 차지하는 총 요소 개수(청크 크기)를 계산합니다. (How)
  let chunk_size = out_dim_size * stride;
  
  // 현재 1차원 인덱스가 어떤 배치(상위 차원들)에 속하는지 계산합니다. (How)
  let batch_idx = idx / chunk_size;
  // 현재 청크(chunk) 내에서 몇 번째 인덱스인지를 구합니다. (나머지 연산)
  let rem = idx % chunk_size;
  // 현재 청크 내에서 결합 축을 기준으로 몇 번째 위치에 있는지를 구합니다. (How)
  let dim_idx = rem / stride;
  // 결합 축보다 하위에 있는 차원에서 몇 번째 위치(stride_idx)인지를 구합니다.
  let stride_idx = rem % stride;
  
  // 현재 계산된 결합 축 상의 위치(dim_idx)가 텐서 A의 크기보다 작은지 검사합니다. (What)
  // 이 조건이 참이면 현재 요소는 텐서 A에서 가져와야 함을 의미합니다. (Why)
  if (dim_idx < a_dim) {
    // 텐서 A 배열 내부에서의 정확한 1차원 원본 인덱스를 복원 계산합니다. (How)
    // 배치 크기 * (A 차원 크기 * 스트라이드) + (A 안에서의 축 위치 * 스트라이드) + 하위 차원 오프셋
    let a_index = batch_idx * (a_dim * stride) + dim_idx * stride + stride_idx;
    // 계산된 인덱스를 사용해 텐서 A의 값을 결과 텐서에 복사합니다.
    out[idx] = a[a_index];
  } else {
    // dim_idx가 a_dim 이상이면 현재 요소는 텐서 B에서 가져와야 합니다.
    // 텐서 B의 내부 차원 인덱스로 변환하기 위해 A가 차지했던 크기를 뺍니다. (How)
    let b_dim_idx = dim_idx - a_dim;
    // 텐서 B 배열 내부에서의 원본 위치를 계산합니다. (How)
    let b_index = batch_idx * (b_dim * stride) + b_dim_idx * stride + stride_idx;
    // 계산된 인덱스를 사용해 텐서 B의 값을 결과 텐서에 복사합니다.
    out[idx] = b[b_index];
  }
}
`;

    /**
     * 파일 생성: 2026-08-12 12:23:09
     * 수정 내역:
     * - 2026-08-12 12:23:09: Docs: Build Apache-style docs and unify tests (fc28607f9d46845175a9bdaf0e9e8c44bace5ecb)
     */
    const WHERE_WGSL = `
// 구조체: Params
// 목적: 조건부 분기(Where) 연산에 사용되는 메타데이터를 저장합니다.
// 작동 방식: 처리할 전체 요소 크기(size)와 패딩 값들을 통해 16바이트 정렬된 메모리 구조를 형성합니다.
struct Params {
  // 변수: size
  // 목적: 배열의 전체 요소 수를 지정합니다.
  // 작동 방식: 커널에서 각 스레드가 유효한 범위 내에 있는지 확인하는 데 사용됩니다.
  size: u32,
  // 변수: workgroups_x
  // 목적: X축 방향 워크그룹의 총 개수입니다.
  // 작동 방식: 글로벌 인덱스 변환 시 X축 길이를 곱하는 계수로 사용됩니다.
  workgroups_x: u32,
  // 변수: pad2 ~ pad7
  // 목적: 메모리 정렬(Alignment)을 맞추기 위한 여유 공간(패딩)들입니다.
  // 작동 방식: WGSL 유니폼 버퍼의 레이아웃 규칙에 맞추기 위해 사용됩니다.
  pad2: u32,
  pad3: u32,
  pad4: u32,
  pad5: u32,
  pad6: u32,
  pad7: u32,
};

// 변수: params
// 목적: 외부에서 제공되는 파라미터 구조체를 바인딩합니다.
// 작동 방식: 바인딩 0에 읽기 전용으로 매핑됩니다.
@group(0) @binding(0) var<uniform> params: Params;

// 변수: cond
// 목적: 요소별로 어느 값을 선택할지 결정하는 조건(Condition) 배열입니다.
// 작동 방식: 값이 0보다 크면 참(True), 그렇지 않으면 거짓(False)으로 평가됩니다.
@group(0) @binding(1) var<storage, read> cond: array<f32>;

// 변수: x
// 목적: 조건이 참(True)일 때 선택될 데이터 배열입니다.
// 작동 방식: 바인딩 2에 할당됩니다.
@group(0) @binding(2) var<storage, read> x: array<f32>;

// 변수: y
// 목적: 조건이 거짓(False)일 때 선택될 데이터 배열입니다.
// 작동 방식: 바인딩 3에 할당됩니다.
@group(0) @binding(3) var<storage, read> y: array<f32>;

// 변수: out
// 목적: 조건에 따라 x 또는 y에서 선택된 결과가 저장될 배열입니다.
// 작동 방식: 바인딩 4에 할당되어 계산 결과를 기록합니다.
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

// 함수: main
// 목적: cond 배열의 값에 따라 병렬로 x 또는 y의 요소를 선택하여 out 배열에 씁니다 (TensorFlow/PyTorch의 where 함수와 유사).
// 작동 방식: 각 스레드가 조건 평가를 거쳐 선택한 값을 기록합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: num_elements
  // 목적: 연산할 전체 배열 요소의 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 읽어옵니다.
  let num_elements = params.size;

  // 변수: workgroups_x
  // 목적: X축 방향의 워크그룹 개수입니다.
  // 작동 방식: 유니폼 버퍼에서 읽어옵니다.
  let workgroups_x = params.workgroups_x;

  // 변수: idx
  // 목적: 현재 스레드가 담당하는 배열 요소의 1차원 인덱스입니다.
  // 작동 방식: 2차원 워크그룹 배열 인덱스를 1차원으로 평면화하여 계산합니다.
  let idx = global_id.x + global_id.y * workgroups_x * 64u;

  // 제어문: if
  // 목적: 배열의 유효 범위를 넘어가는 스레드가 메모리에 접근하지 않게 합니다.
  // 작동 방식: 인덱스가 전체 크기 이상이면 함수를 끝냅니다.
  if (idx >= num_elements) {
    return;
  }

  // 제어문: if-else
  // 목적: 조건에 맞게 x 배열과 y 배열 중 하나의 값을 선택합니다.
  // 작동 방식: cond[idx]가 0보다 크면 x[idx]를, 아니면 y[idx]를 out[idx]에 할당합니다.
  if (cond[idx] > 0.0) {
    out[idx] = x[idx];
  } else {
    out[idx] = y[idx];
  }
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:59:35 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     */
    const DROPOUT_WGSL = `
/**
 * @struct Params
 * @brief 드롭아웃(Dropout) 연산을 수행하기 위해 필요한 메타데이터를 저장하는 구조체입니다. (What)
 * 과적합(Overfitting) 방지를 위해 무작위로 뉴런(값)을 0으로 끄는 확률(p)과 난수 시드(seed) 정보를 GPU에 전달하기 위해 사용됩니다. (Why)
 */
struct Params {
  // 드롭아웃을 적용할 전체 데이터 원소의 개수입니다.
  num_elements: u32,
  // 난수 생성의 기반이 되는 시드(seed) 값입니다. 매번 다른 패턴의 드롭아웃을 적용하기 위해 외부에서 주입됩니다.
  seed: f32,
  // 드롭아웃 확률(p)입니다. (0.0 ~ 1.0) 이 확률보다 낮게 난수가 나오면 해당 값을 0으로 끕니다.
  p: f32,
  // 데이터 정렬 규칙(16바이트)을 충족시키기 위해 존재하는 의미 없는 패딩 값입니다.
  padding: f32,
}

// params: 드롭아웃 파라미터를 담고 있는 Uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// x: 입력 데이터를 보관하고 있는 텐서(1차원 배열)입니다. (읽기 전용)
@group(0) @binding(1) var<storage, read> x: array<f32>;
// out: 드롭아웃 적용 이후 결과가 저장될 출력 데이터 텐서입니다.
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

/**
 * @function pcg_hash
 * @brief PCG (Permuted Congruential Generator) 기반의 해시 함수를 통해 32비트 정수형 난수를 생성합니다. (What)
 * 셰이더 내부에는 내장된 난수 생성기가 없으므로, 인덱스와 시드를 바탕으로 빠르고 균일하게 의사 난수(Pseudo Random)를 만들기 위해 고안되었습니다. (Why)
 * @param input 난수의 입력이 되는 시드 역할의 부호 없는 정수입니다. (How)
 * @return 해시 변환된 새로운 32비트 난수(u32)를 반환합니다.
 */
fn pcg_hash(input: u32) -> u32 {
    // 입력된 정수에 큰 소수를 곱하고 상수를 더해 초기 상태(state)를 섞습니다. (How)
    var state = input * 747796405u + 2891336453u;
    // 비트 시프트 연산과 XOR을 통해 비트 패턴을 비선형적으로 한 번 더 혼합합니다. (How)
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    // 최종적으로 비트 이동 후 XOR하여 고품질의 난수를 반환합니다.
    return (word >> 22u) ^ word;
}

/**
 * @function rand_f32
 * @brief 32비트 정수 형태의 난수를 0.0 이상 1.0 미만의 부동소수점(float) 형태로 정규화합니다. (What)
 * 드롭아웃 확률(p)과 직접 크기를 비교하기 위해 0~1 사이의 값이 필요하기 때문입니다. (Why)
 * @param hash pcg_hash로부터 전달받은 32비트 무작위 정수입니다.
 * @return 0.0과 1.0 사이로 매핑된 실수 난수입니다. (How)
 */
fn rand_f32(hash: u32) -> f32 {
    // 32비트 정수의 최대값(4294967295)으로 나누어 0~1 범위의 실수로 변환합니다. (How)
    return f32(hash) / 4294967295.0;
}

/**
 * @function main
 * @brief 스레드별로 난수를 발생시켜 지정된 확률 p 미만이면 0을, 그 이상이면 스케일링된 원본 값을 출력 텐서에 기록합니다. (What)
 * 신경망 학습 시 특정 노드에 과도하게 의존하는 현상을 막기 위해(Why) 병렬 스레드를 이용하여 고속으로 무작위 노드 비활성화를 수행합니다.
 * @param global_id 워크그룹 및 스레드의 3차원 전역 인덱스입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2차원(Y) 워크그룹 구조를 1차원으로 풀어 현재 스레드의 전역 선형 인덱스를 계산합니다. (How)
    // 참고로 65535u는 X방향 워크그룹의 최대 한계를 가정하여 하드코딩된 변환 값입니다.
    let index = global_id.x + global_id.y * 65535u * 64u;
    
    // 계산된 인덱스가 전체 텐서의 원소 수보다 크거나 같으면 실행을 즉시 중단합니다. (What)
    // 올바르지 않은 메모리 범위를 건드리지 않도록 차단하는 역할입니다. (Why)
    if (index >= params.num_elements) {
        return;
    }
    
    // 현재 인덱스와 외부에서 입력받은 시드를 조합(스케일업)하여 해시 생성기의 입력(input)을 구성하고 난수 정수를 만듭니다. (How)
    let hash = pcg_hash(index + u32(params.seed * 10000.0));
    // 정수 형태의 해시를 0.0 ~ 1.0 사이의 실수 난수로 변환합니다. (How)
    let rand = rand_f32(hash);
    
    // 생성된 난수가 설정된 드롭아웃 확률 p보다 작은지 검사합니다. (What)
    if (rand < params.p) {
        // 확률 분포에 걸렸을 경우(노드 비활성화), 해당 인덱스의 출력값을 0.0으로 만듭니다. (How)
        out[index] = 0.0;
    } else {
        // 확률 분포에 걸리지 않은 경우, 원본 데이터를 그대로 유지하되 기대값을 보존하기 위해 1/(1-p) 만큼 스케일링(Scaling)하여 저장합니다. (How)
        // 이는 Inverted Dropout 기법으로, 테스트 단계에서 별도의 스케일링 작업 없이 바로 모델을 쓸 수 있게 만들기 위함입니다. (Why)
        out[index] = x[index] * (1.0 / (1.0 - params.p));
    }
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:59:35 +0900 (commit 67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
     * 수정 이력:
     * - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     */
    const MAXPOOL2D_WGSL = `
/**
 * 이 구조체(Params)는 2D 맥스 풀링(Max Pooling 2D) 연산에 필요한 하이퍼파라미터 및 차원 정보를 담고 있습니다.
 * 입력 이미지의 배치, 채널, 크기 정보와 커널 크기, 스트라이드, 패딩 값을 전달하기 위해 존재합니다.
 */
struct Params {
    batch: u32, // 배치 크기입니다.
    channels: u32, // 입력 텐서의 채널 수입니다.
    in_h: u32, // 원본 입력 이미지의 높이입니다.
    in_w: u32, // 원본 입력 이미지의 너비입니다.
    out_h: u32, // 계산되어 출력될 이미지의 높이입니다.
    out_w: u32, // 계산되어 출력될 이미지의 너비입니다.
    kH: u32, // 풀링 커널(필터)의 높이입니다.
    kW: u32, // 풀링 커널(필터)의 너비입니다.
    sH: u32, // 높이 방향의 스트라이드(보폭)입니다.
    sW: u32, // 너비 방향의 스트라이드(보폭)입니다.
    pH: u32, // 높이 방향에 추가된 제로 패딩 크기입니다.
    pW: u32, // 너비 방향에 추가된 제로 패딩 크기입니다.
}

@group(0) @binding(0) var<uniform> params: Params; // GPU에 메타데이터를 공급하는 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // NCHW 형태의 입력 데이터 텐서입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 풀링 결과가 저장될 NCHW 형태의 출력 텐서입니다.

/**
 * main 함수는 합성곱 신경망(CNN)의 핵심 구성 요소인 2D 맥스 풀링 연산을 수행합니다.
 * 이미지의 국소 영역(커널 크기)에서 최댓값만을 추출하여 공간적 차원(Spatial Dimension)을 축소하기 위해 존재합니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // 출력 텐서에서 현재 스레드가 담당할 1D 위치(인덱스)입니다.
    // 연산이 필요한 전체 출력 데이터의 개수를 계산합니다.
    let total = params.batch * params.channels * params.out_h * params.out_w;
    
    // 할당된 스레드 인덱스가 유효 범위를 벗어나면 연산을 중단합니다.
    if (idx >= total) {
        return;
    }
    
    // 1D 인덱스에서 NCHW 포맷에 따라 출력 좌표 (ow, oh, c, b)를 역산합니다.
    let ow = idx % params.out_w; // 출력 맵의 너비(x) 좌표입니다.
    let oh = (idx / params.out_w) % params.out_h; // 출력 맵의 높이(y) 좌표입니다.
    let c = (idx / (params.out_w * params.out_h)) % params.channels; // 채널 인덱스입니다.
    let b = idx / (params.out_w * params.out_h * params.channels); // 배치 인덱스입니다.
    
    // 스트라이드와 패딩을 적용하여 입력 이미지 기준 시작 좌표를 계산합니다.
    let h_start = i32(oh * params.sH) - i32(params.pH);
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    // 최댓값 비교를 위한 초기값을 부동소수점 표현 가능한 가장 작은 값으로 설정합니다.
    var max_val = -3.402823466e+38; // -FLT_MAX
    
    // 커널의 높이(kH)와 너비(kW) 영역을 순회하며 최댓값을 찾기 위한 이중 루프입니다.
    for (var kh = 0u; kh < params.kH; kh++) {
        for (var kw = 0u; kw < params.kW; kw++) {
            // 커널 내 오프셋을 더하여 실제 입력 데이터 상의 좌표를 구합니다.
            let h = h_start + i32(kh);
            let w = w_start + i32(kw);
            
            // 계산된 좌표가 이미지 경계 안쪽에 있는지(유효한 데이터인지) 검사합니다.
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                // NCHW 포맷에 따른 입력 텐서의 1D 메모리 인덱스를 계산합니다.
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                let val = input[in_idx]; // 입력값을 읽어옵니다.
                
                // 기존의 max_val과 비교하여 더 큰 값이면 갱신합니다.
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // 커널 영역 전체에서 발견한 최댓값을 출력 텐서의 현재 인덱스에 저장합니다.
    output[idx] = max_val;
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:59:35 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     */
    const AVGPOOL2D_WGSL = `
/**
 * @struct Params
 * @brief 2D 평균 풀링(Average Pooling 2D) 연산에 필요한 하이퍼파라미터 및 텐서 차원 정보를 저장합니다. (What)
 * 셰이더 내에서 입력 텐서의 특정 영역을 순회하고 평균을 계산하기 위한 기준 값들로 사용됩니다. (Why)
 */
struct Params {
    // 배치(batch) 크기입니다. 여러 이미지를 동시에 처리하기 위한 차원입니다.
    batch: u32,
    // 채널(channel) 수입니다. 예를 들어 RGB 이미지의 경우 3이 될 수 있습니다.
    channels: u32,
    // 입력 이미지의 높이(height) 차원 크기입니다.
    in_h: u32,
    // 입력 이미지의 너비(width) 차원 크기입니다.
    in_w: u32,
    // 출력 이미지의 높이 차원 크기입니다. 연산 후의 공간적 크기를 나타냅니다.
    out_h: u32,
    // 출력 이미지의 너비 차원 크기입니다.
    out_w: u32,
    // 풀링 커널(kernel)의 높이 크기입니다.
    kH: u32,
    // 풀링 커널(kernel)의 너비 크기입니다.
    kW: u32,
    // 높이 방향의 스트라이드(stride, 이동 보폭)입니다.
    sH: u32,
    // 너비 방향의 스트라이드(stride, 이동 보폭)입니다.
    sW: u32,
    // 높이 방향의 패딩(padding) 크기입니다.
    pH: u32,
    // 너비 방향의 패딩(padding) 크기입니다.
    pW: u32,
}

// params: 연산 정보를 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// input: 풀링 연산을 수행할 원본 입력 데이터 배열(읽기 전용)입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>;
// output: 풀링 연산 결과가 기록될 출력 데이터 배열입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

/**
 * @function main
 * @brief 컴퓨트 셰이더의 진입점으로, 각 스레드가 하나의 출력 픽셀에 대한 2D 평균 풀링 연산을 수행합니다. (What)
 * GPU의 수많은 스레드를 활용하여 이미지 전체 영역 및 배치 데이터를 병렬로 압축 처리하기 위해 존재합니다. (Why)
 * @param global_id 워크그룹 및 스레드의 전역적인 3차원 위치(인덱스)입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 현재 스레드의 선형 인덱스를 가져옵니다.
    let idx = global_id.x;
    
    // 계산해야 할 전체 출력 요소의 총합을 구합니다 (배치 * 채널 * 출력높이 * 출력너비). (How)
    let total = params.batch * params.channels * params.out_h * params.out_w;
    
    // 스레드 인덱스가 유효 범위를 벗어나면 즉시 함수를 종료(return)하여 잘못된 메모리 접근을 막습니다. (Why)
    if (idx >= total) {
        return;
    }
    
    // 1차원 인덱스 idx를 4차원 좌표 (b, c, oh, ow)로 변환하는 과정입니다. (How)
    // 현재 픽셀이 속한 출력 이미지의 너비 위치(ow)를 구합니다.
    let ow = idx % params.out_w;
    // 현재 픽셀이 속한 출력 이미지의 높이 위치(oh)를 구합니다.
    let oh = (idx / params.out_w) % params.out_h;
    // 현재 픽셀이 속한 채널 위치(c)를 구합니다.
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    // 현재 픽셀이 속한 배치 위치(b)를 구합니다.
    let b = idx / (params.out_w * params.out_h * params.channels);
    
    // 입력 이미지에서 현재 커널이 적용될 시작 Y좌표(높이)를 계산합니다. 패딩을 고려하여 음수가 될 수도 있습니다. (What)
    let h_start = i32(oh * params.sH) - i32(params.pH);
    // 입력 이미지에서 현재 커널이 적용될 시작 X좌표(너비)를 계산합니다. 패딩을 고려합니다.
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    // 풀링 영역 내의 픽셀 값들을 누적하기 위한 합계 변수입니다. (What)
    var sum = 0.0;
    // 풀링 영역 내에서 실제로 유효한 픽셀의 개수를 셉니다. (경계 밖은 제외하기 위함) (Why)
    var count = 0.0;
    
    // 커널의 높이만큼 반복하여 수직 방향 픽셀들을 순회합니다. (How)
    for (var kh = 0u; kh < params.kH; kh++) {
        // 커널의 너비만큼 반복하여 수평 방향 픽셀들을 순회합니다. (How)
        for (var kw = 0u; kw < params.kW; kw++) {
            // 현재 순회 중인 픽셀의 실제 입력 텐서상 Y좌표입니다.
            let h = h_start + i32(kh);
            // 현재 순회 중인 픽셀의 실제 입력 텐서상 X좌표입니다.
            let w = w_start + i32(kw);
            
            // 유효성 검사: 계산된 (h, w)가 이미지 경계를 벗어나지 않는지(0 이상, 입력 크기 미만) 확인합니다. (What)
            // 패딩 영역이나 이미지 범위를 넘어간 곳의 값은 무시하여 올바른 평균을 구하기 위함입니다. (Why)
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                // 4차원 좌표 (b, c, h, w)를 다시 1차원 인덱스(in_idx)로 변환합니다. (How)
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                
                // 해당 입력 픽셀의 값을 합산 변수에 누적시킵니다.
                sum += input[in_idx];
                // 유효한 픽셀을 한 개 처리했으므로 카운트를 증가시킵니다.
                count += 1.0;
            }
        }
    }
    
    // 유효한 픽셀 카운트가 1개 이상일 경우 정상적으로 평균을 계산합니다. (What)
    // 0으로 나누기(Division by zero) 오류를 방지하기 위함입니다. (Why)
    if (count > 0.0) {
        // 총 누적 합(sum)을 유효 픽셀 개수(count)로 나누어 평균을 구한 후, 출력 배열의 1차원 인덱스에 저장합니다. (How)
        output[idx] = sum / count;
    } else {
        // 유효한 픽셀이 전혀 없었다면(예: 모두 패딩 영역인 경우) 결과값을 0으로 처리합니다. (How)
        output[idx] = 0.0;
    }
}
`;

    /**
     * 파일 생성일: 2026-08-12 12:59:35 +0900 (commit 67c4ce9901dbb7caf2710e9ad03514f48956cfa6)
     * 수정 이력:
     * - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     */
    const IM2COL_WGSL = `
/**
 * 이 구조체(Params)는 이미지 데이터(공간적 텐서)를 열(Column) 기반 행렬로 변환하기 위한 컨볼루션 인자들을 담고 있습니다.
 * 입력 이미지의 크기, 커널(필터)의 크기, 스트라이드, 패딩 등 im2col 연산에 필수적인 하이퍼파라미터를 제공하기 위해 존재합니다.
 */
struct Params {
  N: u32, // 배치 크기 (Batch size)입니다.
  C: u32, // 입력 채널 수 (Channels)입니다.
  H: u32, // 입력 이미지의 원본 높이 (Height)입니다.
  W: u32, // 입력 이미지의 원본 너비 (Width)입니다.
  K_h: u32, // 커널(필터)의 높이입니다.
  K_w: u32, // 커널(필터)의 너비입니다.
  stride: u32, // 합성곱 연산 시 필터가 이동하는 보폭(스트라이드)입니다.
  padding: u32, // 입력 이미지 가장자리에 추가할 제로 패딩의 크기입니다.
  H_out: u32, // 연산 후 생성될 출력 특성 맵의 높이입니다.
  W_out: u32, // 연산 후 생성될 출력 특성 맵의 너비입니다.
};

@group(0) @binding(0) var<uniform> params: Params; // GPU에 컨볼루션 설정값을 전달하는 유니폼 버퍼입니다.
@group(0) @binding(1) var<storage, read> input: array<f32>; // NCHW 형태로 펼쳐진 원본 이미지 입력 배열입니다.
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // 변환된 행렬 형태의 데이터가 기록될 출력 배열입니다.

/**
 * main 함수는 합성곱(Convolution) 연산을 행렬 곱(MatMul)으로 효율적으로 수행하기 위해
 * 이미지 데이터의 국소적 패치(Local patch)를 추출하여 2D 행렬 형태로 재배치(im2col)합니다.
 * 이를 통해 GPU 상에서 고속의 GEMM(General Matrix Multiply) 라이브러리 및 최적화를 활용할 수 있습니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x; // 출력 im2col 버퍼에서의 현재 스레드의 1D 인덱스입니다.
  // 변환될 출력 배열의 총 요소 개수를 계산합니다 (N * H_out * W_out * C * K_h * K_w).
  let num_elements = params.N * params.H_out * params.W_out * params.C * params.K_h * params.K_w;
  
  // 계산할 인덱스가 배열 크기를 넘어가면 실행을 종료합니다.
  if (idx >= num_elements) { return; }

  var temp = idx; // 1차원 인덱스를 다차원 인덱스로 역계산하기 위해 임시 변수에 저장합니다.
  
  // 출력 버퍼의 인덱스에서 채널 및 커널 위치에 해당하는 차원 값을 추출합니다.
  let c_kw_kh = temp % (params.C * params.K_h * params.K_w);
  temp = temp / (params.C * params.K_h * params.K_w); // 다음 차원 추출을 위해 값을 나눕니다.
  
  // 출력 특성 맵의 공간적 위치(높이, 너비) 차원 값을 추출합니다.
  let h_out_w_out = temp % (params.H_out * params.W_out);
  temp = temp / (params.H_out * params.W_out); // 다음 차원 추출을 위해 값을 나눕니다.
  
  // 최종적으로 배치(Batch) 인덱스를 추출합니다.
  let n = temp % params.N;

  // 커널 내에서의 로컬 x, y 좌표 및 채널 인덱스를 계산합니다.
  let k_w = c_kw_kh % params.K_w; // 커널 내에서의 너비 인덱스
  let k_h = (c_kw_kh / params.K_w) % params.K_h; // 커널 내에서의 높이 인덱스
  let c = c_kw_kh / (params.K_w * params.K_h); // 입력 채널 인덱스

  // 출력 특성 맵 내에서의 x, y 좌표를 계산합니다.
  let w_out = h_out_w_out % params.W_out; // 출력 맵에서의 너비 위치
  let h_out = h_out_w_out / params.W_out; // 출력 맵에서의 높이 위치

  // 커널 위치와 스트라이드, 패딩을 고려하여 원본 입력 이미지 상의 실제 y, x 좌표를 역산합니다.
  let h_in = i32(h_out * params.stride) - i32(params.padding) + i32(k_h);
  let w_in = i32(w_out * params.stride) - i32(params.padding) + i32(k_w);

  // 계산된 원본 위치가 이미지 경계 내부인지 검사합니다.
  if (h_in >= 0 && h_in < i32(params.H) && w_in >= 0 && w_in < i32(params.W)) {
    // 경계 내부라면 NCHW 포맷에 따라 입력 배열의 1D 인덱스를 계산하고 값을 가져와 저장합니다.
    let in_idx = ((n * params.C + c) * params.H + u32(h_in)) * params.W + u32(w_in);
    output[idx] = input[in_idx];
  } else {
    // 경계 밖이라면 패딩 영역이므로 0.0을 채워 넣습니다.
    output[idx] = 0.0;
  }
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:59:35 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     */
    const COL2IM_WGSL = `
/**
 * @struct Params
 * @brief 합성곱(Convolution) 연산의 역전파 과정에서 필요한 col2im (Column to Image) 연산용 파라미터 구조체입니다. (What)
 * im2col을 통해 펼쳐진 행렬 형태의 그레이디언트를 다시 원래 텐서(이미지) 형태로 복원하기 위한 정보를 담고 있습니다. (Why)
 */
struct Params {
  // 배치 크기 (Batch size)
  N: u32,
  // 채널의 개수 (Channels)
  C: u32,
  // 원본 입력 텐서의 높이 (Height)
  H: u32,
  // 원본 입력 텐서의 너비 (Width)
  W: u32,
  // 합성곱 커널의 높이 크기
  K_h: u32,
  // 합성곱 커널의 너비 크기
  K_w: u32,
  // 필터 이동 보폭 (Stride)
  stride: u32,
  // 텐서 테두리에 덧붙인 패딩 크기
  padding: u32,
  // 합성곱 연산 결과 출력 텐서의 높이
  H_out: u32,
  // 합성곱 연산 결과 출력 텐서의 너비
  W_out: u32,
};

// params: col2im 역산 및 복원 계산을 위한 각종 텐서 차원들을 포함한 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// grad_x_col: im2col 형태로 전개되어 있던 그레이디언트 1차원 배열입니다 (읽기 전용).
@group(0) @binding(1) var<storage, read> grad_x_col: array<f32>;
// grad_x: 다시 원래 이미지 크기(N, C, H, W)로 합산 복원될 입력 텐서에 대한 그레이디언트 배열입니다.
@group(0) @binding(2) var<storage, read_write> grad_x: array<f32>;

/**
 * @function main
 * @brief 컴퓨트 셰이더의 메인 함수로, 원본 이미지의 픽셀 인덱스별로 연관되었던 모든 커널 윈도우들의 기울기(gradient)를 합산(accumulate)합니다. (What)
 * CNN 합성곱 층에서 입력값에 대한 역전파(Backpropagation)를 수행하여 가중치 갱신에 필요한 값을 도출하기 위해 (Why) 작성되었습니다.
 * 
 * @param global_id 워크그룹 내 스레드의 3차원 전역 인덱스입니다.
 */
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 현재 스레드가 처리할 원본 텐서 상의 1차원 인덱스입니다. (How)
  let idx = global_id.x;
  // 전체 요소 개수 = 배치 * 채널 * 높이 * 너비 를 계산합니다.
  let num_elements = params.N * params.C * params.H * params.W;
  
  // 인덱스가 전체 크기를 벗어나면 즉시 함수를 빠져나가(return) 오류를 막습니다. (Why)
  if (idx >= num_elements) { return; }

  // 1차원 인덱스 idx를 4차원 좌표인 (n, c, h, w)로 복원하기 위한 임시 변수입니다. (How)
  var temp = idx;
  // 너비 차원 (Width) 복원
  let w = temp % params.W;
  temp = temp / params.W;
  // 높이 차원 (Height) 복원
  let h = temp % params.H;
  temp = temp / params.H;
  // 채널 차원 (Channel) 복원
  let c = temp % params.C;
  // 배치 차원 (Batch) 복원
  let n = temp / params.C;

  // 원본 텐서의 특정 픽셀 (n, c, h, w)에 모여들 그레이디언트 값을 누적하기 위한 실수형 변수입니다. (What)
  var val = 0.0;
  
  // 커널의 높이(K_h)만큼 반복하며 이 픽셀에 영향을 주었던 합성곱 윈도우들을 역추적합니다. (How)
  for (var k_h = 0u; k_h < params.K_h; k_h = k_h + 1u) {
    // 패딩이 적용된 높이 좌표를 계산합니다. (What)
    let h_plus_pad = h + params.padding;
    
    // 현재 커널 인덱스 k_h보다 크거나 같은지 검사하여 필터 범위를 벗어나지 않았는지 판단합니다. (Why)
    if (h_plus_pad >= k_h) {
      // 커널 내부에서의 오프셋을 제거하여 원본 인덱스를 역계산합니다. (How)
      let h_rem = h_plus_pad - k_h;
      // 스트라이드(stride) 조건에 맞게 정확하게 나누어 떨어지는 윈도우 위치인지 검사합니다. (What)
      if (h_rem % params.stride == 0u) {
        // 출력 텐서 상의 y좌표(h_out)를 복원 계산합니다.
        let h_out = h_rem / params.stride;
        // 계산된 출력 좌표가 실제 출력 텐서의 높이 범위 내에 있는지 검사합니다.
        if (h_out < params.H_out) {
          
          // 커널의 너비(K_w)만큼 반복하며 수평 방향 윈도우들을 탐색합니다. (How)
          for (var k_w = 0u; k_w < params.K_w; k_w = k_w + 1u) {
            // 패딩이 적용된 너비 좌표를 계산합니다.
            let w_plus_pad = w + params.padding;
            
            // 현재 커널 인덱스 k_w보다 크거나 같은지 확인하여 유효 범위인지 검사합니다.
            if (w_plus_pad >= k_w) {
              // 커널 너비 내의 오프셋을 제거합니다.
              let w_rem = w_plus_pad - k_w;
              // 수평 스트라이드 조건에 정확히 부합하는지 확인합니다. (What)
              if (w_rem % params.stride == 0u) {
                // 출력 텐서 상의 x좌표(w_out)를 계산합니다. (How)
                let w_out = w_rem / params.stride;
                // 계산된 출력 좌표가 실제 출력 텐서 너비 범위에 들어오는지 검증합니다.
                if (w_out < params.W_out) {
                  // 배치 번호는 원본과 동일하게 가져옵니다.
                  let n_out = n;
                  // 출력 평면 2D 상의 1차원 선형 인덱스(hw_out)를 계산합니다. (How)
                  let hw_out = h_out * params.W_out + w_out;
                  // 커널 안에서의 채널 및 2D 윈도우 인덱스(c_kw_kh)를 1차원으로 계산합니다.
                  let c_kw_kh = (c * params.K_h + k_h) * params.K_w + k_w;
                  
                  // 위에서 계산한 값들을 바탕으로, 2차원으로 전개되었던 grad_x_col 배열의 실제 접근 인덱스를 합성합니다. (What)
                  let col_idx = (n_out * (params.H_out * params.W_out) + hw_out) * (params.C * params.K_h * params.K_w) + c_kw_kh;
                  // 전개된 배열에서 가져온 기울기(gradient) 값을 현재 픽셀의 누적기(val)에 더합니다. (How)
                  val = val + grad_x_col[col_idx];
                }
              }
            }
          }
          
        }
      }
    }
  }

  // 역추적된 윈도우들로부터 누적 계산이 모두 끝난 총 그레이디언트 값을 출력 배열에 저장합니다. (What)
  grad_x[idx] = val;
}
`;

    /**
     * 생성일: 확인 불가 (Git 기록 없음 혹은 커밋 대기 상태)
     * 수정 이력:
     * - 특이사항 없음
     */
    const PERMUTE_WGSL = `
// 구조체: Params
// 역할 (WHAT): Permute(전치/축 교환) 연산에 필요한 차원, 형상(Shape) 및 보폭(Stride) 정보를 담은 구조체입니다.
// 목적 (WHY): 입력 텐서의 축을 지정된 순서대로 재배열하여 출력 텐서의 메모리 레이아웃을 계산하기 위해 유니폼 데이터를 전달합니다.
// 동작 방식 (HOW): rank와 총 요소 수를 제공하고, 최대 8차원을 지원하기 위해 vec4 두 개를 이어서 strides와 shape 정보를 제공합니다.
struct Params {
  // 변수: rank
  // 역할: 텐서가 가진 총 차원의 수
  rank: u32,
  // 변수: numElements
  // 역할: 텐서 내 존재하는 전체 데이터 요소의 개수
  numElements: u32,
  // 변수: pad1, pad2
  // 역할: 16바이트 메모리 정렬을 위한 패딩 변수
  pad1: u32,
  pad2: u32,
  // 변수: in_strides
  // 역할: 입력 텐서의 첫 4차원(0~3)에 대한 메모리 보폭
  in_strides: vec4<u32>,
  // 변수: in_strides_ext
  // 역할: 입력 텐서의 확장 4차원(4~7)에 대한 메모리 보폭
  in_strides_ext: vec4<u32>,
  // 변수: out_shape
  // 역할: 출력 텐서의 첫 4차원(0~3)에 대한 크기(Shape)
  out_shape: vec4<u32>,
  // 변수: out_shape_ext
  // 역할: 출력 텐서의 확장 4차원(4~7)에 대한 크기(Shape)
  out_shape_ext: vec4<u32>,
  // 변수: out_strides
  // 역할: 출력 텐서의 첫 4차원(0~3)에 대한 메모리 보폭
  out_strides: vec4<u32>,
  // 변수: out_strides_ext
  // 역할: 출력 텐서의 확장 4차원(4~7)에 대한 메모리 보폭
  out_strides_ext: vec4<u32>,
};

// 변수: params
// 역할: 셰이더 전역에서 접근 가능한 Permute 연산용 메타데이터 유니폼 버퍼
@group(0) @binding(0) var<uniform> params: Params;

// 변수: input
// 역할: 원본 데이터가 들어 있는 읽기 전용 스토리지 버퍼
@group(0) @binding(1) var<storage, read> input: array<f32>;

// 변수: output
// 역할: 축이 변환된 최종 데이터가 저장될 쓰기 가능한 스토리지 버퍼
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 함수: main
// 역할 (WHAT): 출력 인덱스를 기반으로 다차원 좌표를 복원하고, 이를 입력 텐서의 보폭과 매칭하여 축 교환된 값을 저장합니다.
// 목적 (WHY): 텐서의 차원 순서를 바꾸는 연산(예: 행렬 전치, 채널 축 변경)을 GPU를 활용하여 병렬로 빠르게 수행하기 위함입니다.
// 동작 방식 (HOW): 각 스레드는 2D 기반 ID를 통해 출력 인덱스(out_idx)를 얻고, 반복문을 통해 각 차원별 인덱스를 분리해내어, 원본 보폭과 곱하여 in_idx를 도출해 값을 복사합니다.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 변수: out_idx
  // 역할: 2D 그리드 디스패치(dispatch)를 지원하기 위해 글로벌 ID x, y를 결합하여 만든 1차원 출력 인덱스
  // 주석: Compute global index supporting 2D grid dispatch
  let out_idx = global_id.x + global_id.y * 65535u * 64u;
  
  // 조건문: 데이터 범위 초과 검사
  // 역할: 스레드의 계산된 인덱스가 전체 요소 크기를 넘어서는지 판단하여 초과 시 연산을 중지합니다.
  if (out_idx >= params.numElements) {
    return;
  }
  
  // 변수: out_idx_remaining
  // 역할: 각 차원의 좌표를 구하기 위해 나누기/나머지 연산을 하면서 변해가는 임시 나머지 인덱스 값
  var out_idx_remaining = out_idx;
  // 변수: in_idx
  // 역할: 입력 텐서에서 실제 데이터를 읽어올 1차원 메모리 인덱스의 누적 값
  var in_idx = 0u;
  
  // 반복문: for 루프 (모든 차원에 대한 순회)
  // 역할 (WHAT): 최상위 차원부터 시작하여 현재 차원에 해당하는 좌표를 구하고, 이를 바탕으로 원래 입력 배열의 인덱스를 계산합니다.
  // 목적 (WHY): 다차원 구조가 평면 배열(flat array)로 선형화되어 있으므로, 출력의 구조를 풀어 입력의 구조로 맵핑해야 하기 때문입니다.
  // 동작 방식 (HOW): 0부터 rank-1까지 순회하면서 차원(i)에 맞는 보폭(Stride) 값을 가져오고, 좌표(coord)를 구한 후 입력 인덱스를 누적합니다.
  for (var i = 0u; i < params.rank; i = i + 1u) {
    // 변수: out_stride
    // 역할: 현재 루프 차원(i)에 해당하는 출력 텐서의 보폭
    var out_stride = 0u;
    // 변수: in_stride
    // 역할: 현재 루프 차원(i)에 해당하는 입력 텐서의 보폭
    var in_stride = 0u;
    
    // 조건문: 차원(i) 확인 및 보폭 할당
    // 역할 (WHAT): 루프 인덱스 i 값에 따라 vec4에 묶여 있는 각 차원의 보폭 값을 가져옵니다.
    // 목적 (WHY): WGSL에서는 배열 인덱싱을 지원하지 않는 vec4 구조체 필드에 동적으로 접근하기 위해 하드코딩 된 조건 분기가 필요하기 때문입니다.
    // 동작 방식 (HOW): i가 0~7 중 어느 것인지 확인하고, 해당하는 x, y, z, w 컴포넌트 값을 보폭 변수에 저장합니다.
    if (i == 0u) { out_stride = params.out_strides.x; in_stride = params.in_strides.x; }
    else if (i == 1u) { out_stride = params.out_strides.y; in_stride = params.in_strides.y; }
    else if (i == 2u) { out_stride = params.out_strides.z; in_stride = params.in_strides.z; }
    else if (i == 3u) { out_stride = params.out_strides.w; in_stride = params.in_strides.w; }
    else if (i == 4u) { out_stride = params.out_strides_ext.x; in_stride = params.in_strides_ext.x; }
    else if (i == 5u) { out_stride = params.out_strides_ext.y; in_stride = params.in_strides_ext.y; }
    else if (i == 6u) { out_stride = params.out_strides_ext.z; in_stride = params.in_strides_ext.z; }
    else if (i == 7u) { out_stride = params.out_strides_ext.w; in_stride = params.in_strides_ext.w; }
    
    // 변수: coord
    // 역할: 남은 1차원 인덱스를 출력 보폭으로 나누어 얻은 현재 차원(i)의 논리적 좌표값
    let coord = out_idx_remaining / out_stride;
    
    // 변수: out_idx_remaining 갱신
    // 역할: 다음 차원 계산을 위해 현재 차원에서 처리된 부분을 제외한 나머지(나머지 연산)를 저장합니다.
    out_idx_remaining = out_idx_remaining % out_stride;
    
    // 변수: in_idx 누적
    // 역할: 도출된 논리적 좌표(coord)에 원래 텐서의 보폭(in_stride)을 곱해, 원본 텐서에서 데이터를 읽어올 정확한 1차원 메모리 주소를 누적해 나갑니다.
    in_idx = in_idx + coord * in_stride;
  }
  
  // 변수 output 배열 쓰기
  // 역할: 매핑된 입력 텐서의 1차원 인덱스 위치(in_idx)에 있는 데이터를 읽어와 출력 텐서 위치(out_idx)에 복사하여 위치 바꿈(permute)을 완료합니다.
  output[out_idx] = input[in_idx];
}
`;

    /**
     * 생성일 (Created): 2026-08-12 12:59:35 +0900
     * 수정 내역 (Modified):
     *   - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     */
    const BATCHED_MATMUL_WGSL = `
/**
 * @struct Params
 * @brief 배치 행렬 곱셈(Batched Matrix Multiplication)을 제어하기 위한 행렬의 차원 크기와 스트라이드(stride) 정보를 저장합니다. (What)
 * 입력 행렬 텐서 A와 B의 형태(M, N, K)와 연속적인 배치 접근을 위한 메모리 오프셋을 계산할 때 사용하기 위해 정의되었습니다. (Why)
 */
struct Params {
  // 배치(Batch)의 개수입니다. 한 번에 여러 쌍의 행렬 곱셈을 병렬 처리하기 위한 차원입니다.
  B: u32,
  // 결과 행렬(C)과 왼쪽 행렬(A)의 행(Row) 개수입니다.
  M: u32,
  // 결과 행렬(C)과 오른쪽 행렬(B)의 열(Column) 개수입니다.
  N: u32,
  // 왼쪽 행렬(A)의 열 개수이자 오른쪽 행렬(B)의 행 개수로, 내적(Dot product)이 이루어지는 공통 차원의 길이입니다.
  K: u32,
  // 왼쪽 행렬(A)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideA: u32,
  // 오른쪽 행렬(B)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideB: u32,
  // 결과 행렬(C)에서 다음 배치로 넘어가기 위해 필요한 원소의 개수(보폭)입니다.
  strideC: u32,
};

// params: 배치 크기 및 행렬 차원 정보를 GPU 스레드들에게 제공하는 uniform 버퍼입니다.
@group(0) @binding(0) var<uniform> params: Params;
// a: 첫 번째(왼쪽) 입력 행렬 데이터들을 담고 있는 1차원 배열(읽기 전용)입니다.
@group(0) @binding(1) var<storage, read> a: array<f32>;
// b: 두 번째(오른쪽) 입력 행렬 데이터들을 담고 있는 1차원 배열(읽기 전용)입니다.
@group(0) @binding(2) var<storage, read> b: array<f32>;
// c: 행렬 곱셈의 결과가 저장될 출력 배열(읽기/쓰기 가능)입니다.
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

/**
 * @function main
 * @brief 주어진 배치(Batch)에 대해 행렬 A와 B의 내적을 수행하여 행렬 C의 각 요소를 계산합니다. (What)
 * 어텐션 메커니즘 등 신경망 구조에서 다중 배치의 텐서를 한 번에 곱하기 위해 (Why) 3차원 그리드로 병렬 실행됩니다.
 * 
 * @param global_id 워크그룹과 스레드의 3차원 인덱스입니다. (x: 열(Column), y: 행(Row), z: 배치(Batch)를 나타냅니다.) (How)
 */
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  // 스레드의 x 인덱스로, 연산할 결과 행렬 C의 열(Column) 위치를 할당합니다.
  let col = global_id.x;
  // 스레드의 y 인덱스로, 연산할 결과 행렬 C의 행(Row) 위치를 할당합니다.
  let row = global_id.y;
  // 스레드의 z 인덱스로, 현재 처리할 배치(Batch) 번호를 할당합니다.
  let batch = global_id.z;

  // 할당된 인덱스들이 지정된 행렬 크기나 배치 수를 초과하는지 검사합니다. (What)
  // 워크그룹 크기(8x8)로 인해 남는 스레드가 유효하지 않은 메모리에 접근하는 것을 방지하기 위함입니다. (Why)
  if (row >= params.M || col >= params.N || batch >= params.B) {
    return;
  }

  // 1차원 배열 A에서 현재 배치의 현재 행이 시작되는 오프셋을 계산합니다. (How)
  let a_offset = batch * params.strideA + row * params.K;
  // 1차원 배열 B에서 현재 배치의 현재 열이 시작되는 오프셋을 계산합니다.
  let b_offset = batch * params.strideB + col;
  // 1차원 결과 배열 C에서 현재 배치의 위치(row, col)에 해당하는 저장 인덱스를 계산합니다.
  let c_offset = batch * params.strideC + row * params.N + col;

  // 내적(Dot product)을 누적하기 위한 실수형 변수를 선언하고 0으로 초기화합니다. (What)
  var sum: f32 = 0.0;
  
  // 공통 차원인 K번만큼 반복하여 행렬 A의 특정 행과 행렬 B의 특정 열의 요소들을 곱하고 더합니다. (How)
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    // 행렬 A에서는 열(k) 방향으로 이동하고, 행렬 B에서는 행(k) 방향으로 이동(B의 행 길이인 N만큼 점프)하면서 값을 곱하여 sum에 누적시킵니다. (How)
    sum = sum + a[a_offset + k] * b[b_offset + k * params.N];
  }

  // 계산된 내적 최종 결과(sum)를 출력 배열 C의 오프셋 위치에 저장합니다. (What)
  c[c_offset] = sum;
}
`;

    /**
     * Created: 2026-08-12T12:14:52+09:00
     * Modified:
     *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
     *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
     *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * gpuCore.ts — GPU 코어 API (초기화, 텐서 생명주기, 개별 op 디스패치)
     *
     * H-01 Fix: 모든 op에 _globalPipelineCache 적용 (셰이더 재컴파일 방지)
     * NH-03 Fix: quota를 maxStorageBufferBindingSize 기반으로 설정 (maxBufferSize는 단일 버퍼 크기 제한이지 VRAM 용량이 아님)
     * NH-01 Fix: 개별 op 함수들을 internal로 유지, pyodideBridge에서는 executeGraph만 노출
     * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
     * ARC-01 Fix: device.pushErrorScope로 OOM 감지 시도
     * L-01 Fix: dispatchKernel 헬퍼로 모든 op의 반복 코드 통합 (DRY)
     */
    /**
     * WHAT: 모든 WGSL 셰이더 코드를 커널 이름에 매핑하여 저장하는 전역 읽기 전용 레지스트리 맵입니다.
     * WHY: 런타임에 셰이더 코드를 이름으로 조회하고 파이프라인 캐시 초기화 시 한 번에 반영하기 위해 존재합니다.
     * HOW: Map 객체를 생성하여 문자열 키와 WGSL 코드 문자열 값을 쌍으로 저장합니다.
     */
    const KERNEL_REGISTRY = new Map([
        ['matmul', MATMUL_WGSL],
        ['batched_matmul', BATCHED_MATMUL_WGSL],
        ['relu', RELU_WGSL],
        ['add', ADD_WGSL],
        ['mul', MUL_WGSL],
        ['transpose', TRANSPOSE_WGSL],
        ['relu_backward', RELU_BACKWARD_WGSL],
        ['sub', SUB_WGSL],
        ['neg', NEG_WGSL],
        ['div', DIV_WGSL],
        ['exp', EXP_WGSL],
        ['log', LOG_WGSL],
        ['sigmoid', SIGMOID_WGSL],
        ['tanh', TANH_WGSL],
        ['sigmoid_backward', SIGMOID_BACKWARD_WGSL],
        ['tanh_backward', TANH_BACKWARD_WGSL],
        ['fill', FILL_WGSL],
        ['sum', SUM_WGSL],
        ['max', MAX_WGSL],
        ['sum_axis', SUM_AXIS_WGSL],
        ['axpy', AXPY_WGSL],
        ['pad', PAD_WGSL],
        ['gather', GATHER_WGSL],
        ['scatter', SCATTER_WGSL],
        ['cat', CAT_WGSL],
        ['where', WHERE_WGSL],
        ['dropout', DROPOUT_WGSL],
        ['maxpool2d', MAXPOOL2D_WGSL],
        ['avgpool2d', AVGPOOL2D_WGSL],
        ['im2col', IM2COL_WGSL],
        ['col2im', COL2IM_WGSL],
        ['permute', PERMUTE_WGSL],
    ]);
    // VUL-001 Fix: Register kernel names automatically to keep whitelist in sync
    registerKernelNames(KERNEL_REGISTRY.keys());
    /**
     * WHAT: CPU로 읽어오기 위해 대기 중인 GPU 스테이징 버퍼들을 추적하는 전역 맵입니다.
     * WHY: 비동기 맵핑(mapAsync)이 완료된 버퍼를 기록해 두고 나중에 동기적으로 데이터를 읽어올 수 있게 하기 위해 필요합니다.
     * HOW: 텐서 핸들(문자열)을 키로, 매핑된 GPUBuffer와 AllocationToken 객체를 값으로 유지합니다.
     */
    const _pendingStagingBuffers = new Map();
    const _inFlightMapPromises = new Map();
    /**
     * WHAT: GPU 코어의 런타임 메모리와 모든 캐시된 리소스를 초기화(해제)하는 함수입니다.
     * WHY: 디바이스 유실(Device Lost) 이벤트가 발생하거나 시스템 강제 리셋 시 남은 자원의 메모리 누수를 방지하기 위해 존재합니다.
     * HOW: 텐서 레지스트리, 쿼터 매니저, 파이프라인 캐시를 지우고, 대기 중인 스테이징 버퍼들도 순회하여 언맵(unmap) 및 파괴(destroy)합니다.
     */
    function resetRuntimeMemory() {
        _globalRegistry.clear();
        _globalQuotaManager.reset();
        _globalPipelineCache.clear(); // L-03 Fix: device lost 시 파이프라인 캐시도 무효화
        _inFlightMapPromises.clear();
        /**
         * WHAT: 스테이징 버퍼 맵에 남아있는 모든 엔트리를 순회하여 파괴하는 루프입니다.
         * WHY: 사용되지 않고 남겨진 스테이징 버퍼가 VRAM을 계속 차지하는 것을 방지하기 위해 필요합니다.
         * HOW: _pendingStagingBuffers 맵의 모든 값을 하나씩 꺼내어 언맵 및 소각을 시도하고 토큰을 해제합니다.
         */
        for (const [, obj] of _pendingStagingBuffers) {
            try {
                obj.stagingBuffer.unmap();
            }
            catch { /* already unmapped */ }
            try {
                obj.stagingBuffer.destroy();
            }
            catch { /* already destroyed */ }
            _globalQuotaManager.releaseToken(obj.token);
        }
        _pendingStagingBuffers.clear();
    }
    /**
     * WHAT: 시스템 로거가 존재할 경우 로그 메시지를 남기는 래퍼 함수입니다.
     * WHY: 글로벌 환경(예: Pyodide)에 주입된 로그 함수가 있을 때만 호출하여 콘솔 오염을 막고 안전한 디버깅을 하기 위함입니다.
     * HOW: globalThis에서 log 함수를 찾아 존재하면 호출하고 오류 발생 시 조용히 무시(catch)합니다.
     */
    function _safeLog(msg) {
        try {
            if (typeof globalThis.log === 'function') {
                globalThis.log(msg, 'system');
            }
        }
        catch (e) { }
    }
    /**
     * WHAT: WebGPU 하위 시스템을 초기화하고 메모리 한도 설정 및 셰이더 컴파일을 수행하는 비동기 진입점 함수입니다.
     * WHY: 텐서 연산을 수행하기 전에 GPU 디바이스를 획득하고 하드웨어 제약을 파악하며 파이프라인을 준비하기 위해 필수적입니다.
     * HOW: initWebGPU를 호출하여 디바이스를 얻고, 디바이스 어댑터의 limits를 조회하여 메모리 할당 한도를 설정한 뒤, 모든 커널을 사전 컴파일(warmup)합니다.
     */
    async function init(options) {
        _safeLog(`[gpuCore.ts] init started`);
        setDeviceLostCallback(() => {
            resetRuntimeMemory();
        });
        try {
            _safeLog(`[gpuCore.ts] calling initWebGPU...`);
            await initWebGPU(options);
            _safeLog(`[gpuCore.ts] initWebGPU finished`);
        }
        catch (e) {
            _safeLog(`[gpuCore.ts] initWebGPU threw error: ${e.message}`);
            throw e;
        }
        // NH-03: 실제 GPU 제한 조회 후 쿼터 조정
        /**
         * WHAT: 초기화된 WebGPU 어댑터 객체입니다.
         * WHY: 현재 시스템 GPU의 하드웨어 한계(limits)와 기능 정보를 파악하여 안전한 메모리 할당량을 계산하기 위해 조회합니다.
         * HOW: getAdapter() 함수를 호출하여 가져옵니다.
         */
        const adapter = getAdapter();
        if (adapter) {
            /**
             * WHAT: 현재 GPU 어댑터가 지원하는 하드웨어 제약사항을 담은 객체입니다.
             * WHY: 버퍼 바인딩 크기나 컴퓨트 워크그룹 크기의 안전 한계선을 알기 위해 참조합니다.
             * HOW: adapter.limits 프로퍼티를 통해 가져옵니다.
             */
            const limits = adapter.limits;
            if (limits.maxComputeWorkgroupSizeX < 64) {
                console.warn(`[AMEVA] Warning: Device maxComputeWorkgroupSizeX (${limits.maxComputeWorkgroupSizeX}) is less than 64. Kernels are optimized for 64.`);
            }
            /**
             * WHAT: 스토리지 버퍼가 단일 바인딩 시 사용할 수 있는 최대 바이트 크기입니다.
             * WHY: 이 값을 기준으로 사용 가능한 전체 VRAM 용량을 간접적으로 추정하기 위해 필요합니다.
             * HOW: limits.maxStorageBufferBindingSize를 사용하며, 정보가 없으면 기본값(256MB)으로 설정합니다.
             */
            const maxBinding = limits.maxStorageBufferBindingSize ?? 256 * 1024 * 1024;
            /**
             * WHAT: 사용자가 직접 명시한 VRAM 사용 상한(바이트)입니다.
             * WHY: 시스템의 기본 휴리스틱을 무시하고 사용자 설정에 따라 자원을 제어할 수 있도록 옵션으로 받습니다.
             * HOW: options 인자에서 vramLimitBytes 프로퍼티를 참조합니다.
             */
            const userLimit = options?.vramLimitBytes;
            /**
             * WHAT: 할당할 수 있는 최대 하드 VRAM 한도입니다.
             * WHY: 시스템 메모리 초과를 방지하기 위해 엄격한 상한선을 두기 위해 계산합니다.
             * HOW: 사용자 지정값이 있으면 8GB를 넘지 않는 선에서 채택하고, 없으면 바인딩 크기의 4배와 8GB 중 작은 값을 사용합니다.
             */
            const hardLimit = userLimit
                ? Math.min(userLimit, 8 * 1024 * 1024 * 1024)
                : Math.min(maxBinding * 4, 8 * 1024 * 1024 * 1024); // binding 크기의 4배를 총 VRAM 추정
            /**
             * WHAT: 메모리 압박이 시작될 때 경고를 보내거나 GC를 유도하기 위한 소프트 한도입니다.
             * WHY: 하드 한도에 도달하기 전 선제적인 리소스 회수 타이밍을 잡기 위해 존재합니다.
             * HOW: 하드 한도의 75%로 계산합니다.
             */
            const softLimit = Math.floor(hardLimit * 0.75);
            _globalQuotaManager.setLimits(Math.floor(hardLimit), Math.floor(softLimit));
            console.info(`[AMEVA] GPU quota set: soft=${(softLimit / 1e9).toFixed(2)}GB, ` +
                `hard=${(hardLimit / 1e9).toFixed(2)}GB ` +
                `(maxStorageBindingSize=${(maxBinding / 1e9).toFixed(2)}GB)`);
        }
        // H-NEW-08: 비동기 파이프라인 사전 컴파일
        await warmupKernels();
    }
    /**
     * WHAT: 등록된 모든 커널 셰이더를 WebGPU 컴퓨트 파이프라인으로 사전 컴파일하는 함수입니다.
     * WHY: 실행 시점에 셰이더 컴파일이 발생하여 프레임 드랍이나 실행 지연이 생기는 것을 방지하기 위함입니다.
     * HOW: KERNEL_REGISTRY 맵을 순회하여 각 셰이더 코드와 이름 배열을 추출하고 _globalPipelineCache.warmup()을 호출합니다.
     */
    async function warmupKernels() {
        /**
         * WHAT: KERNEL_REGISTRY에서 추출한 커널 이름(key)과 셰이더 소스코드(wgslCode) 객체의 배열입니다.
         * WHY: 파이프라인 캐시의 warmup 메서드에 한꺼번에 전달할 형식을 맞추기 위해 생성합니다.
         * HOW: Array.from()을 사용하여 맵 엔트리를 배열로 변환한 후 map()으로 객체화합니다.
         */
        const entries = Array.from(KERNEL_REGISTRY.entries()).map(([key, wgslCode]) => ({ key, wgslCode }));
        await _globalPipelineCache.warmup(entries);
    }
    /**
     * WHAT: 핸들에 해당하는 텐서의 메타데이터(크기, 타입, 버퍼 크기 등)를 반환하는 함수입니다.
     * WHY: 파이썬 브릿지나 외부에서 현재 텐서의 형태 정보를 조회해야 할 때 사용됩니다.
     * HOW: 전역 레지스트리에서 핸들로 레코드를 조회한 뒤 TensorInfo 객체를 구성하여 반환합니다.
     */
    function getTensorInfo(handle) {
        /**
         * WHAT: 핸들로 조회된 내부 텐서 레코드 객체입니다.
         * WHY: 저장된 shape, dtype 등의 메타데이터를 추출하기 위해 필요합니다.
         * HOW: _globalRegistry.get(handle)을 호출하여 얻어옵니다.
         */
        const record = _globalRegistry.get(handle);
        return {
            handle: record.handle,
            shape: [...record.shape],
            dtype: record.dtype,
            byteLength: record.byteLength,
            disposed: record.disposed
        };
    }
    /**
     * WHAT: 주어진 텐서의 데이터를 GPU에서 CPU로 비동기적으로 읽어 Float32Array로 반환하는 함수입니다.
     * WHY: 연산 결과가 포함된 GPU 버퍼의 데이터를 사용자나 프레임워크가 확인할 수 있도록 하기 위해 제공됩니다.
     * HOW: 레지스트리에서 버퍼를 조회하고 readBufferToFloat32Array 헬퍼를 사용해 데이터를 복사 후 반환합니다.
     */
    function read(handle) {
        /**
         * WHAT: 핸들로 조회된 텐서 레코드 객체입니다.
         * WHY: 실제 GPUBuffer 참조와 버퍼 길이를 알아내기 위해 필요합니다.
         * HOW: _globalRegistry.get(handle) 호출을 통해 가져옵니다.
         */
        const record = _globalRegistry.get(handle);
        return readBufferToFloat32Array(record.buffer, record.byteLength);
    }
    /**
     * WHAT: 텐서 버퍼의 데이터를 읽기 위해 GPU 메모리를 매핑(map)하는 비동기 함수입니다.
     * WHY: 즉시 읽기(read)와 달리 맵핑과 데이터 복사를 분리하여 제로 카피(Zero Copy)나 스트리밍 최적화를 지원하기 위해 존재합니다.
     * HOW: 레지스트리에서 버퍼를 조회한 뒤 맵핑을 수행하고 반환된 스테이징 버퍼를 _pendingStagingBuffers에 저장합니다.
     */
    async function mapBufferAsync(handle) {
        // If already staged and mapped, return immediately
        if (_pendingStagingBuffers.has(handle)) {
            return;
        }
        // If a mapping operation is already in-flight for this handle, coalesce with existing promise
        const inFlight = _inFlightMapPromises.get(handle);
        if (inFlight) {
            return inFlight;
        }
        const record = _globalRegistry.get(handle);
        const promise = (async () => {
            try {
                const { stagingBuffer, token } = await mapBufferAsync$1(record.buffer, record.byteLength);
                _pendingStagingBuffers.set(handle, { stagingBuffer, token });
            }
            finally {
                _inFlightMapPromises.delete(handle);
            }
        })();
        _inFlightMapPromises.set(handle, promise);
        return promise;
    }
    /**
     * WHAT: 매핑이 완료된 스테이징 버퍼에서 대상 배열로 데이터를 동기 복사하는 함수입니다.
     * WHY: mapBufferAsync 호출 이후 실제 데이터를 사용자의 자바스크립트 버퍼 혹은 Pyodide 메모리로 옮기기 위해 사용됩니다.
     * HOW: _pendingStagingBuffers에서 버퍼를 찾아 실제 대상 배열(outArray)에 복사하고 스테이징 버퍼를 정리합니다.
     */
    function readMappedInto(handle, outArray) {
        /**
         * WHAT: 이전 mapBufferAsync 호출로 준비된 스테이징 버퍼 관련 정보 객체입니다.
         * WHY: 복사해올 실제 소스 버퍼에 접근하기 위해 맵에서 꺼내어 참조합니다.
         * HOW: _pendingStagingBuffers.get(handle)을 통해 조회합니다.
         */
        const obj = _pendingStagingBuffers.get(handle);
        if (!obj) {
            throw new Error(`[AMEVA] No staged buffer for handle "${handle}". Call mapBufferAsync first.`);
        }
        _pendingStagingBuffers.delete(handle);
        /**
         * WHAT: Pyodide나 WebAssembly 환경의 메모리 뷰를 감싸는 프록시 객체입니다.
         * WHY: 외부 WASM 메모리를 다룰 때 버퍼 포인터 획득과 해제를 안전하게 처리하기 위해 변수에 저장합니다.
         * HOW: 초기엔 null로 두고 outArray 타입에 따라 getBuffer() 결과가 할당됩니다.
         */
        let bufProxy = null;
        try {
            /**
             * WHAT: 데이터 복사가 기록될 최종 대상 Float32Array입니다.
             * WHY: 스테이징 버퍼의 데이터를 CPU가 직접 다룰 수 있는 형식으로 전달받기 위해 필요합니다.
             * HOW: bufProxy.data를 통해 참조를 얻거나 outArray 자체를 Float32Array로 캐스팅합니다.
             */
            let actualData;
            if (outArray && typeof outArray.getBuffer === 'function') {
                bufProxy = outArray.getBuffer("f32");
                actualData = bufProxy.data;
            }
            else {
                actualData = outArray;
            }
            // F-009 Fix: 대상 배열 크기와 원본 텐서 크기 검증
            const record = _globalRegistry.get(handle);
            if (actualData.byteLength !== record.byteLength) {
                throw new Error(`[AMEVA Forge] readMappedInto size mismatch. Expected ${record.byteLength} bytes, got ${actualData.byteLength} bytes.`);
            }
            readMappedInto$1(obj.stagingBuffer, obj.token, actualData);
        }
        finally {
            // _readMappedInto already releases the token!
            // H-NEW-06: bufProxy.release() 실패 시에도 리소스 정리 보장
            if (bufProxy) {
                try {
                    bufProxy.release();
                }
                catch { /* ignore */ }
            }
        }
    }
    /**
     * WHAT: 사용을 마친 특정 텐서를 해제하는 함수입니다.
     * WHY: 외부 사용자가 더 이상 텐서 메모리를 사용하지 않을 때 메모리를 GPU에서 해제하기 위해 호출됩니다.
     * HOW: _globalRegistry.dispose()를 호출하여 핸들에 연결된 레코드를 삭제하고 버퍼 소멸 스케줄을 잡습니다.
     */
    function dispose(handle) {
        _globalRegistry.dispose(handle);
    }
    /**
     * WHAT: 단일 WebGPU 컴퓨트 셰이더 커널을 디스패치(실행 요청)하는 공통 헬퍼 함수입니다.
     * WHY: 개별 연산 함수(add, sub 등)에 중복되는 버퍼 바인딩 및 파이프라인 생성 코드를 통합하여 유지보수성을 높이기 위해 존재합니다.
     * HOW: 유니폼 파라미터 버퍼를 생성하고 파이프라인 캐시를 조회한 뒤, 바인드 그룹을 설정하여 컴퓨트 패스를 큐에 제출합니다.
     */
    function dispatchKernel(opts) {
        // NH-07 Fix: shaderGuard에서 커널 이름 검증
        assertAllowedKernelName(opts.opKey);
        /**
         * WHAT: WebGPU 작업을 제출할 대상 논리 디바이스입니다.
         * WHY: 커맨드 인코더 생성과 버퍼 조작을 위해 필요합니다.
         * HOW: getDevice() 함수를 호출하여 가져옵니다.
         */
        const device = getDevice();
        /**
         * WHAT: 셰이더로 전달될 스칼라 인자(크기, 차원 등)를 담는 GPU 유니폼 버퍼입니다.
         * WHY: GPU 셰이더 내에서 텐서 크기 등의 동적인 파라미터를 읽을 수 있어야 연산이 가능하기 때문입니다.
         * HOW: 최소 16바이트 정렬 크기를 만족하도록 디바이스에서 UNIFORM 용도로 할당합니다.
         */
        const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(Math.max(16, opts.paramsData.byteLength), // 최소 16바이트 (WebGPU uniform 정렬)
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'uniform', `dispatchKernel_${opts.opKey}`);
        device.queue.writeBuffer(paramsBuffer, 0, opts.paramsData.buffer);
        // H-01: 파이프라인 캐시에서 조회 (없으면 컴파일 후 캐시)
        /**
         * WHAT: 컴파일이 완료된 WebGPU 컴퓨트 파이프라인 객체입니다.
         * WHY: 셰이더 코드를 기반으로 GPU가 작업을 어떻게 수행해야 하는지 구조를 알고 있어야 하기 때문입니다.
         * HOW: opKey와 wgslCode를 사용하여 _globalPipelineCache에서 가져옵니다.
         */
        const { pipeline } = _globalPipelineCache.getPipeline(opts.opKey, opts.wgslCode);
        /**
         * WHAT: 파이프라인에 바인딩될 리소스들의 배열(유니폼 버퍼, 입력 버퍼들, 출력 버퍼)입니다.
         * WHY: 셰이더의 각 바인딩 슬롯(binding 0, 1, 2...)에 정확한 버퍼를 매핑하기 위해 리스트로 준비합니다.
         * HOW: paramsBuffer를 binding 0에, 입력 버퍼들을 그 다음 순서에, 출력 버퍼를 마지막에 배치하여 구성합니다.
         */
        const entries = [
            { binding: 0, resource: { buffer: paramsBuffer } },
            ...opts.inputBuffers.map((buf, i) => ({
                binding: i + 1,
                resource: { buffer: buf }
            })),
            { binding: opts.inputBuffers.length + 1, resource: { buffer: opts.outBuffer } }
        ];
        /**
         * WHAT: 준비된 entries를 기반으로 셰이더와 런타임 버퍼를 연결해주는 바인드 그룹 객체입니다.
         * WHY: 디바이스 커맨드 패스에 리소스 그룹을 설정하기 위해 필수적입니다.
         * HOW: device.createBindGroup을 통해 파이프라인의 레이아웃과 entries를 결합하여 생성합니다.
         */
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
        /**
         * WHAT: GPU 명령들을 기록하기 위한 커맨드 인코더입니다.
         * WHY: 복사, 컴퓨트 패스 등 여러 GPU 조작을 묶어서 큐에 제출하기 위해 사용됩니다.
         * HOW: device.createCommandEncoder()로 생성합니다.
         */
        const commandEncoder = device.createCommandEncoder();
        /**
         * WHAT: 컴퓨트 연산을 기록하는 패스 인코더입니다.
         * WHY: 파이프라인, 바인드 그룹, 디스패치 워크그룹 수 등을 설정하기 위해 필요합니다.
         * HOW: commandEncoder.beginComputePass()를 호출하여 가져옵니다.
         */
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(opts.dispatchX, opts.dispatchY ?? 1);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
        // params 버퍼는 GPU 제출 완료 후 중앙 allocator를 통해 해제
        void device.queue.onSubmittedWorkDone().then(() => {
            try {
                freeBuffer(paramsBuffer, paramsToken);
            }
            catch { }
        });
    }
    // ─────────────────────────────────────────────────────────────────────────────
    // 개별 op 함수들 (내부 사용, pyodideBridge에서는 executeGraph를 통해서만 접근)
    // NH-01 Note: 이 함수들은 JS 테스트와 직접 호출에서만 사용
    // ─────────────────────────────────────────────────────────────────────────────
    /**
     * WHAT: 무작위 값(0~1)으로 채워진 지정된 형태(shape)의 텐서를 생성하는 함수입니다.
     * WHY: 신경망 가중치 초기화나 테스트 코드에서 임의의 데이터가 필요할 때 사용됩니다.
     * HOW: CPU(자바스크립트) 상에서 Float32Array 배열에 난수를 채우고 allocateBuffer로 얻은 GPU버퍼로 복사하여 레지스트리에 등록합니다.
     */
    function random(shape, dtype = "float32") {
        validateDType(dtype);
        /**
         * WHAT: 텐서의 모든 차원을 곱해 산출된 총 원소의 개수입니다.
         * WHY: 1차원 Float32Array를 얼마나 크게 할당하고 루프를 돌릴지 결정하기 위해 계산됩니다.
         * HOW: validateShape 헬퍼를 통해 모양 검증과 동시에 산출됩니다.
         */
        const elements = validateShape(shape, dtype);
        /**
         * WHAT: CPU 메모리 상에 존재하는 실수 데이터 배열입니다.
         * WHY: GPU로 데이터를 전송하기 전 난수값을 임시로 기록하기 위해 할당합니다.
         * HOW: 원소 수(elements)만큼의 크기로 Float32Array를 생성합니다.
         */
        const data = new Float32Array(elements);
        /**
         * WHAT: 배열의 각 위치를 순회하며 난수를 채우는 반복문입니다.
         * WHY: 텐서 전체를 임의의 값으로 초기화하기 위해 실행됩니다.
         * HOW: i를 0부터 elements 전까지 증가시키며 Math.random() 값을 배열에 대입합니다.
         */
        for (let i = 0; i < elements; i++)
            data[i] = Math.random();
        /**
         * WHAT: 텐서 전체 데이터가 차지할 실제 바이트 크기입니다.
         * WHY: GPU 버퍼를 할당할 때 정확한 메모리 공간 크기가 필요하므로 계산합니다.
         * HOW: Float32 원소 개수에 4(바이트)를 곱합니다.
         */
        const byteLength = elements * 4;
        /**
         * WHAT: GPU 메모리 내에 새로 할당된 버퍼와 추적 토큰입니다.
         * WHY: 텐서 데이터를 영속적으로 저장하고 나중에 사용할 수 있도록 하기 위함입니다.
         * HOW: allocateBuffer 헬퍼를 사용하여 STORAGE, COPY_SRC, COPY_DST 용도로 버퍼를 생성합니다.
         */
        const { buffer, token } = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        writeFloat32Array(buffer, data);
        return _globalRegistry.register({ buffer, token, shape, dtype, byteLength });
    }
    /**
     * WHAT: 기존의 Float32Array 데이터를 GPU 텐서로 업로드(복사)하여 핸들을 반환하는 함수입니다.
     * WHY: 외부 이미지 데이터나 입력 특징(feature) 배열을 GPU 메모리로 올려 연산을 수행할 수 있게 만들기 위해 존재합니다.
     * HOW: Pyodide 버퍼 프록시 혹은 일반 배열 데이터를 기반으로 GPU 버퍼를 할당하고 값을 복사한 후 레지스트리에 등록합니다.
     */
    function uploadFloat32Array(data, shape) {
        /**
         * WHAT: 업로드할 원본 데이터가 복사된 또는 참조된 Float32Array입니다.
         * WHY: WebGPU 버퍼에 쓰기 명령을 수행하려면 반드시 이 형태의 타입화된 배열이어야 하기 때문입니다.
         * HOW: 조건에 따라 bufProxy.data 또는 data 자체를 캐스팅하여 할당합니다.
         */
        let actualData;
        /**
         * WHAT: 외부 WASM 환경(Pyodide 등)에서 제공하는 버퍼 메모리 프록시 객체입니다.
         * WHY: 외부에 노출된 메모리 포인터 접근 후 자원 누수를 막기 위해 명시적인 해제(release)가 필요하기 때문에 변수에 잡아둡니다.
         * HOW: data 객체가 getBuffer 함수를 가지고 있으면 이를 호출하여 초기화하고 아니면 null을 유지합니다.
         */
        let bufProxy = null;
        if (data && typeof data.getBuffer === 'function') {
            bufProxy = data.getBuffer("f32");
            actualData = bufProxy.data;
        }
        else {
            actualData = data;
        }
        /**
         * WHAT: 입력된 형태(shape)가 지녀야 할 원소 총 개수입니다.
         * WHY: 형태 배열과 실제 전달된 배열의 바이트 길이가 일치하는지 검증하기 위해 필요합니다.
         * HOW: validateShape를 호출하며 actualData의 바이트 크기를 넘겨 정합성을 검사합니다.
         */
        const elements = validateShape(shape, "float32", actualData.byteLength);
        /**
         * WHAT: GPU에 할당될 메모리 총 바이트 수입니다.
         * WHY: allocateBuffer 헬퍼에 필요한 바이트 단위를 맞추기 위해 사용됩니다.
         * HOW: 산출된 원소 개수에 4를 곱합니다.
         */
        const byteLength = elements * 4;
        const { buffer, token } = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        writeFloat32Array(buffer, actualData);
        if (bufProxy)
            bufProxy.release();
        return _globalRegistry.register({ buffer, token, shape, dtype: "float32", byteLength });
    }
    /**
     * WHAT: 두 개의 2차원 텐서에 대해 행렬 곱셈(Matmul)을 수행하는 함수입니다.
     * WHY: 신경망의 완전 연결층(Dense Layer)이나 어텐션 매커니즘 등 주요 선형 대수 연산을 지원하기 위해 존재합니다.
     * HOW: 두 텐서의 차원을 검증하고, 결과용 버퍼를 새로 생성한 뒤 matmul 셰이더를 dispatchKernel로 호출합니다.
     */
    function matmul(handleA, handleB) {
        /**
         * WHAT: 첫 번째 입력 행렬(A)의 레코드입니다.
         * WHY: A 행렬의 shape와 GPU 버퍼 포인터를 알아내기 위해 필요합니다.
         * HOW: 전역 레지스트리에서 handleA를 키로 조회합니다.
         */
        const a = _globalRegistry.get(handleA);
        /**
         * WHAT: 두 번째 입력 행렬(B)의 레코드입니다.
         * WHY: B 행렬의 shape와 메모리 버퍼를 확보하여 연산 인자로 쓰기 위해 필요합니다.
         * HOW: 전역 레지스트리에서 handleB로 조회합니다.
         */
        const b = _globalRegistry.get(handleB);
        if (a.shape.length !== 2 || b.shape.length !== 2)
            throw new AMEVAForgeShapeError("Matmul requires 2D tensors");
        if (a.dtype !== "float32" || b.dtype !== "float32")
            throw new AMEVAForgeDTypeError("Matmul requires float32 tensors");
        /**
         * WHAT: A 행렬의 행, A의 열(B의 행), B의 행, B의 열을 나타내는 차원 변수들입니다.
         * WHY: 행렬 곱이 성립하기 위한 내부 차원(K) 일치 여부를 검사하고 워크그룹 수를 계산하기 위함입니다.
         * HOW: 각 텐서의 shape 배열에서 인덱스로 값을 구조 분해하여 할당합니다.
         */
        const M = a.shape[0], K = a.shape[1], K2 = b.shape[0], N = b.shape[1];
        if (K !== K2)
            throw new AMEVAForgeShapeError(`Inner dim mismatch: ${K} != ${K2}`);
        /**
         * WHAT: 결과 행렬(C)이 차지할 총 바이트 크기입니다.
         * WHY: 행렬 곱의 결과 텐서를 저장할 적절한 크기의 GPU 버퍼를 할당하기 위해 계산합니다.
         * HOW: 행 크기(M)와 열 크기(N)를 곱한 값에 float32 크기인 4를 곱합니다.
         */
        const byteLength = M * N * 4;
        const { buffer: cBuffer, token } = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        dispatchKernel({
            opKey: 'matmul',
            wgslCode: MATMUL_WGSL,
            paramsData: new Uint32Array([M, N, K, 0]),
            inputBuffers: [a.buffer, b.buffer],
            outBuffer: cBuffer,
            // M-05: X=col방향=N, Y=row방향=M
            dispatchX: Math.ceil(N / 8),
            dispatchY: Math.ceil(M / 8),
        });
        return _globalRegistry.register({ buffer: cBuffer, token, shape: [M, N], dtype: "float32", byteLength });
    }
    /**
     * WHAT: 주어진 텐서의 모든 원소에 대해 ReLU(Rectified Linear Unit) 활성화 함수를 적용하는 함수입니다.
     * WHY: 신경망에서 음수 값을 제거하여 비선형성을 부여하기 위해 핵심적인 오퍼레이션입니다.
     * HOW: 단일 텐서 버퍼를 읽고, 동일 크기의 출력 버퍼를 만든 후 relu 커널을 디스패치합니다.
     */
    function relu(handle) {
        /**
         * WHAT: 입력 텐서 레코드입니다.
         * WHY: 연산 대상 데이터가 들어있는 GPU 버퍼와 크기를 가져오기 위함입니다.
         * HOW: 레지스트리에서 핸들로 조회합니다.
         */
        const x = _globalRegistry.get(handle);
        if (x.dtype !== "float32")
            throw new AMEVAForgeDTypeError("ReLU requires float32");
        /**
         * WHAT: 입력 텐서 내에 존재하는 실수 요소의 총 개수입니다.
         * WHY: 워크그룹 수를 계산하여 디스패치 크기를 결정하고 셰이더 내에서 배열 경계 검사를 수행하기 위해 필요합니다.
         * HOW: 총 바이트 길이를 4로 나누어 구합니다.
         */
        const numElements = x.byteLength / 4;
        const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        dispatchKernel({
            opKey: 'relu',
            wgslCode: RELU_WGSL,
            paramsData: new Uint32Array([numElements, 0, 0, 0]),
            inputBuffers: [x.buffer],
            outBuffer,
            dispatchX: Math.ceil(numElements / 64),
        });
        return _globalRegistry.register({ buffer: outBuffer, token, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
    }
    /**
     * WHAT: 두 텐서 간의 요소별 덧셈(Element-wise Addition)을 수행하는 함수입니다.
     * WHY: 편향(bias) 더하기, 잔차 연결(residual connection) 등 신경망 연산에서 두 특징 맵을 합칠 때 사용됩니다.
     * HOW: 형태가 같은 두 텐서 버퍼를 넘겨받아 add 셰이더를 실행시키고 새로운 텐서를 생성해 반환합니다.
     */
    function add(handleA, handleB) {
        /**
         * WHAT: 덧셈의 왼쪽 항(A) 텐서 레코드입니다.
         * WHY: A의 버퍼 데이터를 연산 파이프라인에 주입하고 반환 텐서의 모양을 빌리기 위해 참조됩니다.
         * HOW: 레지스트리를 통해 핸들로 가져옵니다.
         */
        const a = _globalRegistry.get(handleA);
        /**
         * WHAT: 덧셈의 오른쪽 항(B) 텐서 레코드입니다.
         * WHY: A와 합쳐질 데이터를 제공하기 위해 조회됩니다.
         * HOW: 레지스트리를 통해 조회됩니다.
         */
        const b = _globalRegistry.get(handleB);
        // F-020 Fix: Direct API (add, mul) exact shape match
        if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i]))
            throw new AMEVAForgeShapeError("Add requires tensors of the exact same shape");
        if (a.dtype !== "float32" || b.dtype !== "float32")
            throw new AMEVAForgeDTypeError("Add requires float32");
        /**
         * WHAT: 두 텐서가 공통으로 가지고 있는 원소의 수입니다.
         * WHY: 셰이더가 병렬로 처리해야 할 작업의 총 개수(스레드 한계)를 지정하기 위함입니다.
         * HOW: 바이트 길이를 4로 나누어 구합니다.
         */
        const numElements = a.byteLength / 4;
        const { buffer: outBuffer, token } = allocateBuffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        dispatchKernel({
            opKey: 'add',
            wgslCode: ADD_WGSL,
            paramsData: new Uint32Array([numElements, 0, 0, 0]),
            inputBuffers: [a.buffer, b.buffer],
            outBuffer,
            dispatchX: Math.ceil(numElements / 64),
        });
        return _globalRegistry.register({ buffer: outBuffer, token, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
    }
    /**
     * WHAT: 두 텐서 간의 요소별 곱셈(Element-wise Multiplication)을 수행하는 함수입니다.
     * WHY: 어텐션 스코어 마스킹이나 활성화된 게이트 통과 등 데이터를 요소별로 가중치와 곱할 때 필요합니다.
     * HOW: 형태가 같은 두 텐서를 기반으로 mul 커널을 디스패치합니다.
     */
    function mul(handleA, handleB) {
        /**
         * WHAT: 곱셈 대상인 첫 번째 텐서(A)의 레코드입니다.
         * WHY: 연산의 피연산자 버퍼로 사용하기 위해 레지스트리에서 가져옵니다.
         * HOW: _globalRegistry.get을 통해 획득합니다.
         */
        const a = _globalRegistry.get(handleA);
        /**
         * WHAT: 곱셈 대상인 두 번째 텐서(B)의 레코드입니다.
         * WHY: 연산의 피연산자 버퍼로 사용하기 위해 레지스트리에서 가져옵니다.
         * HOW: _globalRegistry.get을 통해 획득합니다.
         */
        const b = _globalRegistry.get(handleB);
        // F-020 Fix: Direct API exact shape match
        if (a.shape.length !== b.shape.length || !a.shape.every((v, i) => v === b.shape[i]))
            throw new AMEVAForgeShapeError("Mul requires tensors of the exact same shape");
        if (a.dtype !== "float32" || b.dtype !== "float32")
            throw new AMEVAForgeDTypeError("Mul requires float32");
        /**
         * WHAT: 텐서 내 단일 요소들의 총 개수입니다.
         * WHY: 디스패치할 워크그룹 수를 결정하기 위해 계산됩니다.
         * HOW: 전체 바이트 수를 원소 타입 크기(4)로 나누어 구합니다.
         */
        const numElements = a.byteLength / 4;
        const { buffer: outBuffer, token } = allocateBuffer(a.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        dispatchKernel({
            opKey: 'mul',
            wgslCode: MUL_WGSL,
            paramsData: new Uint32Array([numElements, 0, 0, 0]),
            inputBuffers: [a.buffer, b.buffer],
            outBuffer,
            dispatchX: Math.ceil(numElements / 64),
        });
        return _globalRegistry.register({ buffer: outBuffer, token, shape: [...a.shape], dtype: "float32", byteLength: a.byteLength });
    }
    /**
     * WHAT: 2차원 텐서(행렬)의 행과 열을 뒤집는 전치(Transpose) 연산을 수행하는 함수입니다.
     * WHY: 행렬 곱셈을 수행하기 전에 데이터의 축을 맞추거나 그래디언트 역전파를 위해 텐서를 변형할 때 사용됩니다.
     * HOW: 입력 형태(shape)의 [M, N]을 [N, M]으로 뒤집은 결과를 반환할 출력 버퍼에 기록하도록 transpose 셰이더를 실행합니다.
     */
    function transpose(handle) {
        /**
         * WHAT: 전치 연산을 수행할 원본 2차원 텐서 레코드입니다.
         * WHY: 모양 검증과 입력 버퍼 주입을 위해 필요합니다.
         * HOW: 레지스트리에서 핸들로 텐서 데이터를 조회합니다.
         */
        const x = _globalRegistry.get(handle);
        if (x.shape.length !== 2)
            throw new AMEVAForgeShapeError("Transpose requires 2D tensors");
        if (x.dtype !== "float32")
            throw new AMEVAForgeDTypeError("Transpose requires float32");
        /**
         * WHAT: 원본 텐서의 행 크기(M)와 열 크기(N)입니다.
         * WHY: 전치 후의 새로운 모양을 지정하고, 2차원 워크그룹을 디스패치하기 위해 필요합니다.
         * HOW: x.shape 배열에서 직접 값을 추출합니다.
         */
        const M = x.shape[0], N = x.shape[1];
        const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        dispatchKernel({
            opKey: 'transpose',
            wgslCode: TRANSPOSE_WGSL,
            // F-021 Fix: transpose 커널은 B 파라미터를 요구하므로 기본값 1 전달
            paramsData: new Uint32Array([M, N, 1, 0]),
            inputBuffers: [x.buffer],
            outBuffer,
            // transpose 셰이더: row=global_id.x, col=global_id.y
            dispatchX: Math.ceil(M / 8),
            dispatchY: Math.ceil(N / 8),
        });
        return _globalRegistry.register({ buffer: outBuffer, token, shape: [N, M], dtype: "float32", byteLength: x.byteLength });
    }
    /**
     * WHAT: ReLU 활성화 함수의 도함수(그래디언트)를 계산하여 역전파(Backward)를 수행하는 함수입니다.
     * WHY: 오차 역전파 과정에서 순전파 시 입력값이 0 이상이었던 위치에만 상위 그래디언트를 흘려보내기 위해 필요합니다.
     * HOW: 원본 입력 텐서(x)와 위층에서 전달된 그래디언트 텐서(grad)를 받아, x가 0보다 큰 곳은 grad를, 아니면 0을 출력 버퍼에 씁니다.
     */
    function relu_backward(handleX, handleGrad) {
        /**
         * WHAT: 순전파 때 사용되었던 원래의 입력 텐서(x) 레코드입니다.
         * WHY: 데이터가 양수였는지 음수였는지 판단하는 마스크 역할을 수행하기 위해 조회합니다.
         * HOW: 레지스트리에서 handleX 키로 가져옵니다.
         */
        const x = _globalRegistry.get(handleX);
        /**
         * WHAT: 네트워크 상위 층에서 전파되어 내려온 손실(Loss)의 기울기 텐서입니다.
         * WHY: ReLU의 미분값과 곱해져 현재 층의 최종 기울기를 형성하기 위해 사용됩니다.
         * HOW: 레지스트리에서 handleGrad 키로 가져옵니다.
         */
        const grad = _globalRegistry.get(handleGrad);
        // F-020 Fix: Direct API exact shape match
        if (x.shape.length !== grad.shape.length || !x.shape.every((v, i) => v === grad.shape[i]))
            throw new AMEVAForgeShapeError("ReLU backward: shape mismatch");
        /**
         * WHAT: 그래디언트를 적용할 총 원소의 수입니다.
         * WHY: 워크그룹 수를 계산하여 GPU 전체 스레드 리소스를 할당하기 위해 필요합니다.
         * HOW: 바이트 길이를 4로 나눈 값으로 산출합니다.
         */
        const numElements = x.byteLength / 4;
        const { buffer: outBuffer, token } = allocateBuffer(x.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        dispatchKernel({
            opKey: 'relu_backward',
            wgslCode: RELU_BACKWARD_WGSL,
            paramsData: new Uint32Array([numElements, 0, 0, 0]),
            inputBuffers: [x.buffer, grad.buffer],
            outBuffer,
            dispatchX: Math.ceil(numElements / 64),
        });
        return _globalRegistry.register({ buffer: outBuffer, token, shape: [...x.shape], dtype: "float32", byteLength: x.byteLength });
    }

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
    /**
     * WHAT: 그래프 실행기가 처리할 수 있는 모든 허용된 오퍼레이션(op)의 집합입니다.
     * WHY: 악의적인 JSON 그래프가 알 수 없거나 금지된 셰이더를 실행하여 GPU를 공격하는 것을 방지하기 위한 화이트리스트입니다.
     * HOW: Set 자료구조에 허용되는 오퍼레이션 문자열을 초기화하여 빠른 조회(O(1))를 제공합니다.
     */
    const ALLOWED_OPS = new Set([
        'upload', 'load', 'matmul', 'batched_matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward',
        'sub', 'neg', 'div', 'exp', 'log', 'sigmoid', 'tanh', 'sigmoid_backward', 'tanh_backward',
        'fill', 'sum', 'max', 'sum_axis', 'axpy', 'cat', 'where', 'pad', 'gather', 'scatter', 'maxpool2d', 'avgpool2d',
        'im2col', 'col2im', 'dropout', 'permute', 'matmul_bias_relu'
    ]);
    /**
     * WHAT: 단일 텐서가 가질 수 있는 최대 랭크(차원 수)입니다.
     * WHY: 다차원 반복이나 과도하게 큰 셰이더 파라미터가 유발하는 오버플로우와 성능 저하를 방지하기 위해 제한합니다.
     * HOW: 상수 8로 설정되어, 0(스칼라)부터 8차원까지만 검증을 통과하도록 합니다.
     */
    const MAX_SHAPE_DIM = 8; // NM-06: rank 0~8 허용
    /**
     * WHAT: 단일 텐서가 가질 수 있는 최대 원소의 개수입니다 (float32 기준 1GB).
     * WHY: 악의적인 대용량 텐서 생성 명령으로 인해 브라우저나 디바이스의 VRAM이 고갈(OOM)되는 것을 막기 위함입니다.
     * HOW: 256 * 1024 * 1024 (약 2억 6천만 개)로 정의되어 상한선으로 동작합니다.
     */
    const MAX_ELEMENTS = 256 * 1024 * 1024; // 1GB (float32)
    /**
     * WHAT: 하나의 그래프 실행 요청에 포함될 수 있는 최대 명령어(instruction)의 수입니다.
     * WHY: 너무 거대한 그래프 루프를 실행하다가 메인 스레드가 블로킹되거나 TDR이 발생하는 것을 막습니다.
     * HOW: 상수 10,000으로 설정되어 JSON 배열 길이를 제한합니다.
     */
    const MAX_INSTRUCTIONS = 10_000;
    /**
     * TDR 방지를 위한 워크로드 기반 적응형 분할.
     * WHAT: 단일 커맨드 제출(Submit) 당 누적 허용되는 총 GPU 작업량(원소 수) 예산입니다.
     * WHY: 윈도우 환경 등에서 GPU 작업이 2초 이상 걸리면 발생하는 TDR(Timeout Detection and Recovery)을 회피하기 위해 작업을 쪼갭니다.
     * HOW: 약 1억 개(100M)의 요소를 기준으로 청크(chunk)를 나누도록 상수를 설정합니다.
     */
    const WORKLOAD_BUDGET_ELEMENTS = 100_000_000; // 100M elements per submit
    /**
     * WHAT: 단일 커맨드 제출 당 포함될 수 있는 최대 디스패치(오퍼레이션) 수입니다.
     * WHY: 워크로드가 작더라도 слишком 많은 작은 연산을 한 번에 보내면 발생할 수 있는 오버헤드와 브라우저 블로킹을 방지합니다.
     * HOW: 256개 명령어마다 무조건 큐에 submit 하도록 강제합니다.
     */
    const MAX_OPS_PER_SUBMIT = 256; // 안전장치: element 수 관계없이 256 ops마다 강제 분할
    class GraphTransaction {
        pending = new Map();
        add(record) {
            if (this.pending.has(record.handle)) {
                throw new AMEVAForgeValidationError(`Duplicate pending handle: ${record.handle}`);
            }
            this.pending.set(record.handle, record);
        }
        get(handle) {
            return this.pending.get(handle);
        }
        get handles() {
            return Array.from(this.pending.keys());
        }
        commit(registry) {
            for (const record of this.pending.values()) {
                registry.registerRecord(record);
            }
            this.pending.clear();
        }
        rollback() {
            for (const record of this.pending.values()) {
                try {
                    record.buffer.destroy();
                }
                catch { }
                try {
                    _globalQuotaManager.releaseToken(record.token);
                }
                catch { }
            }
            this.pending.clear();
        }
    }
    /**
     * WHAT: JSON에서 파싱된 단일 명령어 객체의 무결성을 엄격하게 검증하는 함수입니다.
     * WHY: 타입 오류나 범위 초과 등을 가진 악성 데이터가 하위 WebGPU 계층으로 흘러가 충돌을 일으키지 않도록 방어하기 위함입니다.
     * HOW: 속성의 존재 유무와 타입, 배열 길이, 연산의 결과 오버플로우 등을 꼼꼼하게 검사합니다.
     */
    function validateInstruction(inst, idx) {
        if (typeof inst !== 'object' || inst === null) {
            throw new AMEVAForgeSecurityError(`Instruction[${idx}]: must be an object`);
        }
        /**
         * WHAT: 타입 캐스팅을 위해 임시로 생성된 레코드 변수입니다.
         * WHY: unknown 타입을 Record<string, unknown>으로 변환하여 속성에 동적으로 접근하기 위해 필요합니다.
         * HOW: inst를 타입 단언(as)으로 캐스팅합니다.
         */
        const i = inst;
        if (typeof i.op !== 'string') {
            throw new AMEVAForgeSecurityError(`Instruction[${idx}]: op must be a string`);
        }
        if (!ALLOWED_OPS.has(i.op)) {
            throw new AMEVAForgeUnsupportedOpError(`Instruction[${idx}]: unknown op "${i.op}"`);
        }
        if (!Number.isSafeInteger(i.id) || i.id < 1) {
            throw new AMEVAForgeSecurityError(`Instruction[${idx}]: id must be a positive safe integer`);
        }
        if (!Array.isArray(i.shape)) {
            throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape must be an array`);
        }
        // NM-06: rank 0 허용 (스칼라)
        if (i.shape.length > MAX_SHAPE_DIM) {
            throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape rank must be 0–${MAX_SHAPE_DIM}, got ${i.shape.length}`);
        }
        /**
         * WHAT: 해당 명령어 텐서의 누적 원소 수를 계산하는 변수입니다.
         * WHY: 차원의 곱이 안전한 정수 범위를 넘거나 최대 한계(MAX_ELEMENTS)를 초과하는지 확인하기 위해 계산합니다.
         * HOW: 루프를 통해 차원(dim)을 곱하여 누적합니다. 초기값은 스칼라 연산을 위해 1로 시작합니다.
         */
        let elements = 1;
        /**
         * WHAT: shape 배열의 각 차원에 대해 안전성을 검사하는 루프입니다.
         * WHY: 음수 차원, 부동소수점 차원, 정수 오버플로우로 인한 악의적 크기 공격을 차단하기 위해 순회합니다.
         * HOW: for...of 구문으로 각 차원(dim)을 검사하고 elements 변수에 곱합니다.
         */
        for (const dim of i.shape) {
            if (!Number.isSafeInteger(dim) || dim <= 0) {
                throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape dim must be a positive safe integer, got ${dim}`);
            }
            if (dim > Number.MAX_SAFE_INTEGER / elements) {
                throw new AMEVAForgeShapeError(`Instruction[${idx}]: shape product integer overflow`);
            }
            elements *= dim;
        }
        if (elements > MAX_ELEMENTS) {
            throw new AMEVAForgeShapeError(`Instruction[${idx}]: tensor too large (${elements} elements > ${MAX_ELEMENTS})`);
        }
        // NC-06: in 필드가 있으면 배열인지 확인
        if (i.in !== undefined && !Array.isArray(i.in)) {
            throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'in' field must be an array`);
        }
        if (i.params !== undefined && !Array.isArray(i.params)) {
            throw new AMEVAForgeSecurityError(`Instruction[${idx}]: 'params' field must be an array`);
        }
        // F-017 Fix: 각 커널별 엄격한 스키마 검증 (in 개수 및 params 길이 강제)
        const OP_SCHEMA = {
            'upload': { minIn: 0, exactIn: true, minParams: 0, exactParams: true },
            'load': { minIn: 0, exactIn: true, minParams: 0, exactParams: true },
            'fill': { minIn: 0, exactIn: true, minParams: 2, exactParams: true },
            'sum': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'max': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'relu': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'exp': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'log': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'sigmoid': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'tanh': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'neg': { minIn: 1, exactIn: true, minParams: 0, exactParams: true },
            'relu_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'sigmoid_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'tanh_backward': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'pad': { minIn: 1, exactIn: true, minParams: 9, exactParams: true }, // pad는 최대 4차원 36바이트 = 9 uint32s.
            'sum_axis': { minIn: 1, exactIn: true, minParams: 2, exactParams: true },
            'dropout': { minIn: 1, exactIn: true, minParams: 2, exactParams: true },
            'maxpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
            'avgpool2d': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
            'im2col': { minIn: 1, exactIn: true, minParams: 10, exactParams: true },
            'transpose': { minIn: 1, exactIn: true, minParams: 2, exactParams: false },
            'permute': { minIn: 1, exactIn: true, minParams: 1, exactParams: false }, // rank 길이 가변
            'add': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'sub': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'mul': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'div': { minIn: 2, exactIn: true, minParams: 0, exactParams: true },
            'axpy': { minIn: 2, exactIn: true, minParams: 2, exactParams: true },
            'gather': { minIn: 2, exactIn: true, minParams: 7, exactParams: true },
            'scatter': { minIn: 2, exactIn: true, minParams: 7, exactParams: true },
            'matmul': { minIn: 2, exactIn: true, minParams: 3, exactParams: true },
            'batched_matmul': { minIn: 2, exactIn: true, minParams: 4, exactParams: true },
            'where': { minIn: 3, exactIn: true, minParams: 0, exactParams: true },
            'cat': { minIn: 2, exactIn: false, minParams: 1, exactParams: false } // 가변 개수 입력, params는 axis 등
        };
        const opStr = i.op;
        const schema = OP_SCHEMA[opStr];
        if (schema) {
            const inLen = i.in ? i.in.length : 0;
            const pLen = i.params ? i.params.length : 0;
            if (schema.exactIn && inLen !== schema.minIn) {
                throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected exact ${schema.minIn} inputs, got ${inLen}`);
            }
            else if (inLen < schema.minIn) {
                throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected min ${schema.minIn} inputs, got ${inLen}`);
            }
            if (schema.exactParams && pLen !== schema.minParams) {
                throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected exact ${schema.minParams} params, got ${pLen}`);
            }
            else if (pLen < schema.minParams) {
                throw new AMEVAForgeSecurityError(`Instruction[${idx}] op="${opStr}": expected min ${schema.minParams} params, got ${pLen}`);
            }
        }
        // params 타입 검증 (전부 안전한 number 이어야 함)
        if (i.params) {
            for (const p of i.params) {
                if (typeof p !== 'number' || !Number.isFinite(p)) {
                    throw new AMEVAForgeSecurityError(`Instruction[${idx}]: param must be a finite number`);
                }
            }
        }
        return i;
    }
    /**
     * executeGraph — Python 레이지 그래프를 단일 FFI 호출로 GPU에 실행한다.
     * WHAT: Python 등 외부 환경에서 직렬화된 연산 그래프(JSON)를 받아 일괄적으로 GPU에서 실행하는 함수입니다.
     * WHY: 매 연산마다 JS와 WebAssembly/GPU 사이를 왕복(context switch)하면 극심한 오버헤드가 발생하므로, 한 번의 호출로 많은 명령을 처리(Transaction)하기 위해 설계되었습니다.
     * HOW: JSON을 파싱하고, 명령을 검증하며, 적절한 청크로 분할하여 WebGPU 커맨드 버퍼에 기록하고 제출(submit)합니다. 실패 시 트랜잭션을 롤백합니다.
     */
    let _executionQueueChain = Promise.resolve();
    async function executeGraph(instructionsJson, inputs, outputIds) {
        const previous = _executionQueueChain;
        let releaseLock;
        _executionQueueChain = new Promise((resolve) => {
            releaseLock = resolve;
        });
        try {
            await previous;
        }
        catch {
            // Suppress previous transaction error so queue continues processing
        }
        try {
            return await _executeGraphCore(instructionsJson, inputs);
        }
        finally {
            releaseLock();
        }
    }
    async function _executeGraphCore(instructionsJson, jsInputs) {
        // ── 1. Parse ──
        /**
         * WHAT: JSON 문자열에서 파싱된 원시(unvalidated) 자바스크립트 객체 배열입니다.
         * WHY: 외부 문자열 데이터를 자바스크립트 객체 트리로 메모리에 올리기 위해 저장합니다.
         * HOW: JSON.parse()를 시도하며, 예외 발생 시 AMEVAForgeSecurityError를 던집니다.
         */
        let rawInstructions;
        try {
            rawInstructions = JSON.parse(instructionsJson);
        }
        catch {
            throw new AMEVAForgeSecurityError("executeGraph: invalid JSON in instructionsJson");
        }
        if (!Array.isArray(rawInstructions)) {
            throw new AMEVAForgeSecurityError("executeGraph: instructionsJson must be a JSON array");
        }
        if (rawInstructions.length > MAX_INSTRUCTIONS) {
            throw new AMEVAForgeSecurityError(`executeGraph: too many instructions (${rawInstructions.length} > ${MAX_INSTRUCTIONS})`);
        }
        // ── 2. Validate ──
        /**
         * WHAT: 원시 배열을 검증하여 타입 안정성이 보장된 명령어들의 배열입니다.
         * WHY: 이후의 실행 단계(Execution)에서 데이터를 신뢰하고 빠른 연산을 수행할 수 있도록 합니다.
         * HOW: Array.map을 이용해 각 항목을 validateInstruction 함수로 통과시킵니다.
         */
        const instructions = rawInstructions.map(validateInstruction);
        /**
         * WHAT: 그래프 실행 중 'upload' 오퍼레이션에서 사용할 외부 입력 데이터들의 배열입니다.
         * WHY: GPU 밖에서 들어오는 가중치(weights)나 입력 데이터(inputs)를 순차적으로 소비하기 위해 변환해 둡니다.
         * HOW: Pyodide의 toJs()가 있으면 변환하고, 없으면 배열로 간주하며, 둘 다 아니면 빈 배열로 초기화합니다.
         */
        let inputs;
        if (jsInputs && typeof jsInputs.toJs === 'function') {
            inputs = jsInputs.toJs();
        }
        else if (Array.isArray(jsInputs)) {
            inputs = jsInputs;
        }
        else {
            inputs = [];
        }
        // ── 3. Plan ──
        // (In the future: calculate peak memory, check dependency DAG)
        // ── 4. Execute ──
        /**
         * WHAT: WebGPU 작업을 제출할 대상 디바이스 인터페이스입니다.
         * WHY: 커맨드 인코더 생성과 버퍼 조작 및 에러 스코프를 설정하기 위해 필요합니다.
         * HOW: getDevice()를 호출하여 얻습니다.
         */
        const device = getDevice();
        device.pushErrorScope('validation');
        device.pushErrorScope('out-of-memory');
        device.pushErrorScope('internal');
        /**
         * WHAT: 현재 트랜잭션 배치의 GPU 명령을 기록하는 커맨드 인코더 객체입니다.
         * WHY: 개별 오퍼레이션의 상태 변경과 디스패치를 모아 한 번에 GPU 큐로 전송하기 위해 유지합니다.
         * HOW: device.createCommandEncoder()로 생성하며, 청크가 나뉠 때마다 재생성됩니다.
         */
        let commandEncoder = device.createCommandEncoder();
        /**
         * WHAT: 현재 커맨드 인코더에 쌓인 오퍼레이션(디스패치)의 개수입니다.
         * WHY: MAX_OPS_PER_SUBMIT 상한선에 도달했는지 판단하여 강제 플러시(flush)를 트리거하기 위해 카운팅합니다.
         * HOW: 디스패치를 하나 추가할 때마다 1씩 증가시킵니다.
         */
        let opsInCurrentBatch = 0;
        let encoderHasCommands = false;
        /**
         * WHAT: 현재 커맨드 인코더에 제출된 총 연산 원소 수(워크로드)입니다.
         * WHY: WORKLOAD_BUDGET_ELEMENTS 상한선에 도달했는지 확인하여 TDR 현상을 피하도록 배치를 끊기 위해 계산합니다.
         * HOW: 디스패치할 때마다 해당 텐서의 요소를 더합니다.
         */
        let workloadElements = 0;
        /**
         * WHAT: 명령어 ID(명령어별 고유 식별자)를 생성된 텐서 핸들에 매핑하는 객체입니다.
         * WHY: 최종적으로 외부 환경(Python 등)에 어떤 ID가 어떤 텐서를 반환했는지 결과를 돌려주기 위해 유지합니다.
         * HOW: 키는 명령어 id, 값은 TensorHandle(문자열)로 할당합니다.
         */
        const idToHandle = {};
        const idToBuffer = {};
        const idToByteLength = {};
        const transaction = new GraphTransaction();
        let inputIdx = 0;
        /**
         * WHAT: 셰이더의 파라미터 전달을 위해 임시로 생성된 유니폼 버퍼(Uniform Buffer)들의 배열입니다.
         * WHY: GPU 큐 작업이 비동기적으로 완료된 후, 이 임시 버퍼들을 모아서 파괴(destroy)하여 메모리 누수를 방지하기 위해 저장합니다.
         * HOW: 디스패치 과정에서 createBuffer된 파라미터 버퍼들을 push()로 수집합니다.
         */
        const paramsAllocations = [];
        try {
            /**
             * WHAT: 검증된 각 그래프 명령어를 순차적으로 순회하며 GPU 작업으로 변환하는 메인 루프입니다.
             * WHY: 계획된 그래프 연산들을 실제 WebGPU 파이프라인 디스패치로 번역하기 위해 반드시 실행해야 합니다.
             * HOW: for...of 구문을 사용하여 instructions 배열의 각 객체(inst)를 처리합니다.
             */
            for (const inst of instructions) {
                /**
                 * WHAT: 현재 명령어가 결과로 생성할 텐서의 바이트 크기입니다.
                 * WHY: 결과를 담을 출력 버퍼(OutBuffer)의 크기를 GPU에 요청할 때 필요합니다.
                 * HOW: 배열 차원(shape)을 모두 곱한 뒤, float32 크기(4)를 곱하여 계산합니다.
                 */
                const byteLength = inst.shape.reduce((a, b) => a * b, 1) * 4;
                if (inst.op === 'load') {
                    /**
                     * WHAT: load 명령에 전달된 기존 텐서의 핸들 문자열입니다.
                     * WHY: 이미 VRAM에 존재하는 텐서를 그래프의 내부 ID에 매핑하여 입력으로 사용하기 위해 필요합니다.
                     * HOW: inst.handle 속성을 읽어오고 유효성을 검증합니다.
                     */
                    const handle = inst.handle;
                    if (typeof handle !== 'string') {
                        throw new AMEVAForgeSecurityError(`load instruction missing handle`);
                    }
                    if (!_globalRegistry.has(handle)) {
                        console.error(`[GraphExecutor DIAGNOSTIC] load op failed for handle="${handle}". Registered handles count=${_globalRegistry.snapshotHandles().length}, list=${JSON.stringify(_globalRegistry.snapshotHandles())}`);
                    }
                    const record = _globalRegistry.get(handle);
                    // F-018 Fix: JSON 형상과 레지스트리 실제 형상 일치 여부 검사
                    if (inst.shape.length !== record.shape.length || !inst.shape.every((v, i) => v === record.shape[i])) {
                        throw new AMEVAForgeShapeError(`load instruction shape mismatch for handle ${handle}. Expected [${record.shape}], got [${inst.shape}]`);
                    }
                    idToHandle[inst.id] = handle;
                    idToBuffer[inst.id] = record.buffer;
                    idToByteLength[inst.id] = record.byteLength;
                    continue;
                }
                if (inst.op === 'upload') {
                    /**
                     * WHAT: CPU 혹은 Pyodide 메모리에서 건네받은 입력 텐서의 원시 데이터입니다.
                     * WHY: 이 데이터를 GPU 버퍼로 복사하여 연산에 투입하기 위해 필요합니다.
                     * HOW: inputs 배열에서 inputIdx가 가리키는 값을 꺼내옵니다.
                     */
                    const rawData = inputs[inputIdx++];
                    /**
                     * WHAT: 원시 데이터에서 실제 복사 가능한 형태로 추출된 Float32Array 데이터입니다.
                     * WHY: WebGPU의 writeBuffer API는 타입화된 자바스크립트 배열 뷰를 요구하기 때문입니다.
                     * HOW: 데이터의 타입(Pyodide 프록시, Float32Array, 일반 배열)에 따라 변환 및 캐스팅을 수행합니다.
                     */
                    let actualData;
                    /**
                     * WHAT: 외부 WASM 메모리 뷰 프록시입니다.
                     * WHY: 데이터를 다 읽은 후 메모리 락을 해제(release)하기 위해 보존합니다.
                     * HOW: 데이터가 getBuffer 메서드를 제공할 때만 생성됩니다.
                     */
                    let bufProxy = null;
                    if (rawData && typeof rawData.getBuffer === 'function') {
                        bufProxy = rawData.getBuffer("f32");
                        actualData = bufProxy.data;
                    }
                    else if (rawData instanceof Float32Array) {
                        actualData = rawData;
                    }
                    else if (rawData && typeof rawData.toJs === 'function') {
                        const converted = rawData.toJs();
                        actualData = converted instanceof Float32Array ? converted : new Float32Array(converted);
                    }
                    else {
                        throw new AMEVAForgeSecurityError(`upload input[${inputIdx - 1}] is not a Float32Array`);
                    }
                    // VUL-018: NaN / Inf 방어
                    /**
                     * WHAT: 입력 데이터에 무한대나 NaN 값이 포함되어 있는지 검사하고 치환하는 방어 루프입니다.
                     * WHY: NaN 또는 무한대 값이 GPU 셰이더로 흘러가면 연산을 망가뜨리고 TDR 크래시를 유발할 수 있으므로 보호막 역할을 합니다.
                     * HOW: 배열의 모든 요소를 순회하며 Number.isFinite()로 검사하고, 유효하지 않으면 0으로 마스킹합니다.
                     */
                    for (let i = 0; i < actualData.length; i++) {
                        if (!Number.isFinite(actualData[i])) {
                            actualData[i] = 0; // TDR 방지를 위해 0으로 클램프하거나, 경고 로깅 가능 (여기서는 0으로 마스킹)
                            console.warn(`[GraphExecutor] NaN or Inf detected in upload input[${inputIdx - 1}], masked to 0`);
                        }
                    }
                    /**
                     * WHAT: 업로드된 데이터를 담기 위해 GPU에 새로 생성된 스토리지 버퍼와 토큰입니다.
                     * WHY: 데이터를 VRAM으로 옮겨 이후 연산 노드들이 접근할 수 있도록 만듭니다.
                     * HOW: allocateBuffer 헬퍼를 호출하여 STORAGE 및 COPY 용도의 버퍼를 생성합니다.
                     */
                    const { buffer, token } = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'tensor', `Graph_${instructions[0]?.id}`);
                    try {
                        writeFloat32Array(buffer, actualData);
                    }
                    finally {
                        if (bufProxy)
                            bufProxy.release();
                    }
                    const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
                        ? crypto.randomUUID()
                        : Math.random().toString(36).substring(2, 15);
                    const handle = `tensor_${uuid}`;
                    transaction.add({
                        handle,
                        buffer,
                        token,
                        shape: inst.shape,
                        dtype: "float32",
                        byteLength
                    });
                    idToHandle[inst.id] = handle;
                    idToBuffer[inst.id] = buffer;
                    idToByteLength[inst.id] = byteLength;
                    continue;
                }
                assertAllowedKernelName(inst.op);
                let outBuffer;
                if (inst.op === 'axpy') {
                    if (!inst.in || inst.in.length < 2) {
                        throw new AMEVAForgeSecurityError(`Instruction axpy is missing 'in' fields.`);
                    }
                    outBuffer = idToBuffer[inst.in[1]];
                    idToHandle[inst.id] = idToHandle[inst.in[1]];
                    idToBuffer[inst.id] = outBuffer;
                    idToByteLength[inst.id] = byteLength;
                }
                else {
                    const { buffer, token } = allocateBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, 'tensor', `Graph_${instructions[0]?.id}`);
                    outBuffer = buffer;
                    const uuid = typeof crypto !== 'undefined' && crypto.randomUUID
                        ? crypto.randomUUID()
                        : Math.random().toString(36).substring(2, 15);
                    const handle = `tensor_${uuid}`;
                    transaction.add({
                        handle,
                        buffer: outBuffer,
                        token,
                        shape: inst.shape,
                        dtype: "float32",
                        byteLength
                    });
                    idToHandle[inst.id] = handle;
                    idToBuffer[inst.id] = outBuffer;
                    idToByteLength[inst.id] = byteLength;
                }
                /**
                 * WHAT: 현재 오퍼레이션의 유니폼 파라미터를 담기 위해 필요한 바이트 크기입니다.
                 * WHY: 오퍼레이션(패딩, 풀링 등)마다 셰이더가 요구하는 인자의 종류와 개수가 다르므로 가변적인 버퍼 크기를 잡기 위해 결정합니다.
                 * HOW: inst.op 문자열을 판별하여 필요한 바이트 수(최소 32바이트)를 할당합니다.
                 */
                let paramsSize = 32;
                if (inst.op === 'pad')
                    paramsSize = 144;
                else if (inst.op === 'gather' || inst.op === 'scatter')
                    paramsSize = 112;
                else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d')
                    paramsSize = 48;
                else if (inst.op === 'im2col' || inst.op === 'col2im')
                    paramsSize = 40;
                else if (inst.op === 'permute')
                    paramsSize = 112;
                /**
                 * WHAT: GPU 연산 커널에 동적 스칼라 인자를 전달하기 위한 유니폼 버퍼입니다.
                 * WHY: 각 연산의 크기나 특수 인자(스토라이드, 패딩 값 등)를 셰이더 내에서 읽을 수 있게 제공해야 합니다.
                 * HOW: 계산된 paramsSize로 device.createBuffer를 호출하고 UNIFORM 속성을 지정합니다. 생성 후에는 paramsBuffersToDestroy에 등록해 사후 삭제를 예약합니다.
                 */
                const { buffer: paramsBuffer, token: paramsToken } = allocateBuffer(paramsSize, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'uniform', `Graph_${instructions[0]?.id}_params`);
                paramsAllocations.push({ buffer: paramsBuffer, token: paramsToken });
                /**
                 * WHAT: 현재 실행할 WGSL 셰이더 소스 코드를 담는 문자열 변수입니다.
                 * WHY: 오퍼레이션 키워드(inst.op)에 맞는 셰이더를 매핑하여 캐시 조회 및 파이프라인 생성에 넘기기 위함입니다.
                 * HOW: op의 종류에 따라 상수 문자열을 매핑합니다.
                 */
                let wgslCode = "";
                /**
                 * WHAT: 컴퓨트 셰이더를 실행할 3차원 그리드(워크그룹)의 X, Y, Z 디스패치 개수입니다.
                 * WHY: GPU 하드웨어에 얼마나 많은 스레드 블록을 띄워 연산을 처리할지 스케줄링하기 위해 계산합니다.
                 * HOW: 기본값 1로 시작하며, 텐서 크기와 연산 종류에 맞춰 수식이 변동됩니다.
                 */
                let dispatchX = 1, dispatchY = 1, dispatchZ = 1;
                /**
                 * WHAT: 현재 오퍼레이션이 행렬 곱(Matmul) 계열인지 여부를 나타내는 불리언 플래그입니다.
                 * WHY: 행렬 곱 연산은 계산 집약적이므로 TDR 방지를 위해 특별한 청크 단위(chunking) 디스패치가 필요하여 이를 구분하기 위해 사용합니다.
                 * HOW: inst.op가 'matmul'일 때 true로 설정됩니다.
                 */
                let isMatmul = false;
                let B = 1, M = 1, N = 1, K = 1;
                if (inst.op === 'matmul') {
                    if (!inst.params || inst.params.length < 3) {
                        throw new AMEVAForgeSecurityError(`matmul instruction missing params`);
                    }
                    [M, N, K] = inst.params;
                    wgslCode = MATMUL_WGSL;
                    isMatmul = true;
                    // TS-H01 Fix: matmul X축도 65535 클램핑 — 초과분은 Z 차원으로 분산
                    const rawDispatchX = Math.ceil(N / 8);
                    if (rawDispatchX <= 65535) {
                        dispatchX = rawDispatchX;
                    }
                    else {
                        dispatchX = 65535;
                        dispatchZ = Math.ceil(rawDispatchX / 65535);
                    }
                    const maxWorkgroupsM = Math.ceil(M / 8);
                    dispatchY = Math.min(65535, maxWorkgroupsM);
                }
                else if (inst.op === 'batched_matmul') {
                    if (!inst.params || inst.params.length < 4) {
                        throw new AMEVAForgeSecurityError(`batched_matmul instruction missing params`);
                    }
                    const [B_param, N_param, P_param, M_param] = inst.params;
                    B = B_param;
                    wgslCode = BATCHED_MATMUL_WGSL;
                    const rawDispatchX = Math.ceil(P_param / 8);
                    if (rawDispatchX <= 65535) {
                        dispatchX = rawDispatchX;
                    }
                    else {
                        throw new AMEVAForgeSecurityError(`batched_matmul dispatchX exceeded limit: ${rawDispatchX}`);
                    }
                    const rawDispatchY = Math.ceil(N_param / 8);
                    if (rawDispatchY <= 65535) {
                        dispatchY = rawDispatchY;
                    }
                    else {
                        throw new AMEVAForgeSecurityError(`batched_matmul dispatchY exceeded limit: ${rawDispatchY}`);
                    }
                    if (B <= 65535) {
                        dispatchZ = B;
                    }
                    else {
                        throw new AMEVAForgeSecurityError(`batched_matmul dispatchZ (Batch) exceeded limit: ${B}`);
                    }
                    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array(inst.params));
                }
                else if (inst.op === 'transpose') {
                    if (!inst.params || inst.params.length < 2) {
                        throw new AMEVAForgeSecurityError(`transpose instruction missing params`);
                    }
                    const rM = inst.params[0];
                    const rN = inst.params[1];
                    const rB = inst.params.length >= 3 ? inst.params[2] : 1;
                    wgslCode = TRANSPOSE_WGSL;
                    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([rM, rN, rB, 0]));
                    dispatchX = Math.ceil(rM / 8);
                    dispatchY = Math.ceil(rN / 8);
                    dispatchZ = rB;
                }
                else if (inst.op === 'sum_axis') {
                    if (!inst.params || inst.params.length < 2) {
                        throw new AMEVAForgeSecurityError(`sum_axis instruction missing params`);
                    }
                    const [M_param, N_param] = inst.params;
                    wgslCode = SUM_AXIS_WGSL;
                    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M_param, N_param, 0, 0]));
                    dispatchX = Math.ceil(N_param / 64);
                }
                else if (inst.op === 'fill') {
                    if (!inst.params || inst.params.length < 2) {
                        throw new AMEVAForgeSecurityError(`fill instruction missing params`);
                    }
                    const numElements = inst.params[0];
                    const fillValue = inst.params[1];
                    wgslCode = FILL_WGSL;
                    const f32arr = new Float32Array([0, fillValue, 0, 0]);
                    const u32arr = new Uint32Array(f32arr.buffer);
                    u32arr[0] = numElements;
                    device.queue.writeBuffer(paramsBuffer, 0, u32arr);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'axpy') {
                    if (!inst.params || inst.params.length < 2) {
                        throw new AMEVAForgeSecurityError(`axpy instruction missing params`);
                    }
                    const numElements = inst.params[0];
                    const lr = inst.params[1];
                    wgslCode = AXPY_WGSL;
                    const f32arr = new Float32Array([0, lr, 0, 0]);
                    const u32arr = new Uint32Array(f32arr.buffer);
                    u32arr[0] = numElements;
                    device.queue.writeBuffer(paramsBuffer, 0, u32arr);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'pad') {
                    const numElements = byteLength / 4;
                    wgslCode = PAD_WGSL;
                    const p = new Uint32Array(36);
                    /**
                     * WHAT: 패딩 옵션들을 유니폼 버퍼 배열에 복사하는 루프입니다.
                     * WHY: 셰이더에서 사용될 스칼라 인자(정수 및 실수)를 메모리에 연속적으로 배치하기 위해 사용됩니다.
                     * HOW: for 루프를 통해 inst.params 배열의 인자들을 p 배열로 옮기며, 실수형인 패딩 값은 Float32Array 뷰를 통해 씁니다.
                     */
                    for (let i = 0; i < inst.params.length; i++) {
                        if (i === 2)
                            new Float32Array(p.buffer)[2] = inst.params[2];
                        else
                            p[i] = inst.params[i];
                    }
                    device.queue.writeBuffer(paramsBuffer, 0, p);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'gather') {
                    const numElements = byteLength / 4;
                    wgslCode = GATHER_WGSL;
                    const p = new Uint32Array(28);
                    /**
                     * WHAT: 파라미터를 복사하는 짧은 루프입니다.
                     * WHY: gather 커널에 필요한 형태와 인덱싱 오프셋 정보들을 전송하기 위해 복사합니다.
                     * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
                     */
                    for (let i = 0; i < inst.params.length; i++)
                        p[i] = inst.params[i];
                    device.queue.writeBuffer(paramsBuffer, 0, p);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'scatter') {
                    const numElements = inst.params[0];
                    wgslCode = SCATTER_WGSL;
                    const p = new Uint32Array(28);
                    /**
                     * WHAT: scatter 셰이더 인자를 복사하는 루프입니다.
                     * WHY: 분산 배치할 인덱스 스텝 정보를 넘기기 위함입니다.
                     * HOW: 파라미터를 하나씩 Uint32Array에 대입합니다.
                     */
                    for (let i = 0; i < inst.params.length; i++)
                        p[i] = inst.params[i];
                    device.queue.writeBuffer(paramsBuffer, 0, p);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'dropout') {
                    const numElements = byteLength / 4;
                    const seed = inst.params[0];
                    const p_val = inst.params[1];
                    wgslCode = DROPOUT_WGSL;
                    const f32arr = new Float32Array([0, seed, p_val, 0]);
                    const u32arr = new Uint32Array(f32arr.buffer);
                    u32arr[0] = numElements;
                    device.queue.writeBuffer(paramsBuffer, 0, u32arr);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'maxpool2d' || inst.op === 'avgpool2d') {
                    const numElements = byteLength / 4;
                    wgslCode = inst.op === 'maxpool2d' ? MAXPOOL2D_WGSL : AVGPOOL2D_WGSL;
                    const p = new Uint32Array(12);
                    /**
                     * WHAT: 풀링 파라미터를 복사하는 루프입니다.
                     * WHY: 윈도우 크기, 스트라이드, 패딩 등 컨볼루션 구조를 셰이더에 넘기기 위함입니다.
                     * HOW: 요소별로 배열에 대입합니다.
                     */
                    for (let i = 0; i < inst.params.length; i++)
                        p[i] = inst.params[i];
                    device.queue.writeBuffer(paramsBuffer, 0, p);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'im2col' || inst.op === 'col2im') {
                    const numElements = byteLength / 4;
                    wgslCode = inst.op === 'im2col' ? IM2COL_WGSL : COL2IM_WGSL;
                    const p = new Uint32Array(10);
                    /**
                     * WHAT: 공간 변환 파라미터를 복사하는 루프입니다.
                     * WHY: 이미지 크기와 패치 크기 데이터를 셰이더에 전달하기 위해 수행합니다.
                     * HOW: 반복문을 통해 할당합니다.
                     */
                    for (let i = 0; i < inst.params.length; i++)
                        p[i] = inst.params[i];
                    device.queue.writeBuffer(paramsBuffer, 0, p);
                    dispatchX = Math.ceil(numElements / 64);
                }
                else if (inst.op === 'permute') {
                    const numElements = byteLength / 4;
                    wgslCode = PERMUTE_WGSL;
                    const dims = inst.params;
                    const rank = dims.length;
                    const inHandle = idToHandle[inst.in[0]];
                    const inShape = _globalRegistry.get(inHandle).shape;
                    const inStrides = new Array(rank).fill(0);
                    let s = 1;
                    /**
                     * WHAT: 입력 텐서의 각 차원별 메모리 보폭(stride)을 계산하는 역순 루프입니다.
                     * WHY: 다차원 인덱스를 1차원 플랫 메모리 오프셋으로 변환할 때 곱해줄 가중치를 구하기 위해 필요합니다.
                     * HOW: 가장 마지막 차원(우측)부터 시작하여 1부터 차례로 곱해나가며 배열을 채웁니다.
                     */
                    for (let i = rank - 1; i >= 0; i--) {
                        inStrides[i] = s;
                        s *= inShape[i];
                    }
                    const outStrides = new Array(rank).fill(0);
                    let s2 = 1;
                    /**
                     * WHAT: 출력 텐서의 각 차원별 스트라이드를 계산하는 역순 루프입니다.
                     * WHY: 출력을 기록할 1차원 주소를 생성할 때 사용될 가중치를 미리 연산해두기 위함입니다.
                     * HOW: 마찬가지로 맨 우측 차원부터 누적하여 곱합니다.
                     */
                    for (let i = rank - 1; i >= 0; i--) {
                        outStrides[i] = s2;
                        s2 *= inst.shape[i];
                    }
                    const p = new Uint32Array(28);
                    p[0] = rank;
                    p[1] = numElements;
                    /**
                     * WHAT: 계산된 각 차원들의 스트라이드와 형태 정보를 WebGPU vec4 정렬 규칙에 맞게 유니폼 버퍼 패딩 구조에 삽입하는 루프입니다.
                     * WHY: GPU 셰이더 내에서 배열이나 벡터 형태로 데이터를 오차 없이 접근하기 위해 메모리 오프셋을 맞추어 기록합니다.
                     * HOW: i를 0부터 rank 전까지 증가시키며 4개 단위 벡터 위치를 계산하여 씁니다.
                     */
                    for (let i = 0; i < rank; i++) {
                        const vecOffset = i < 4 ? 4 + i : 8 + (i - 4);
                        p[vecOffset] = inStrides[dims[i]];
                        const outShapeOffset = i < 4 ? 12 + i : 16 + (i - 4);
                        p[outShapeOffset] = inst.shape[i];
                        const outStrideOffset = i < 4 ? 20 + i : 24 + (i - 4);
                        p[outStrideOffset] = outStrides[i];
                    }
                    device.queue.writeBuffer(paramsBuffer, 0, p);
                    const totalWorkgroups = Math.ceil(numElements / 64);
                    if (totalWorkgroups <= 65535) {
                        dispatchX = totalWorkgroups;
                    }
                    else {
                        dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
                        dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
                    }
                }
                else if (inst.op === 'sum' || inst.op === 'max') {
                    // Handled entirely dynamically below, but we need to bypass normal flow
                    wgslCode = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
                }
                else {
                    const numElements = byteLength / 4;
                    wgslCode = inst.op === 'relu' ? RELU_WGSL :
                        inst.op === 'add' ? ADD_WGSL :
                            inst.op === 'mul' ? MUL_WGSL :
                                inst.op === 'sub' ? SUB_WGSL :
                                    inst.op === 'neg' ? NEG_WGSL :
                                        inst.op === 'div' ? DIV_WGSL :
                                            inst.op === 'relu_backward' ? RELU_BACKWARD_WGSL :
                                                inst.op === 'exp' ? EXP_WGSL :
                                                    inst.op === 'log' ? LOG_WGSL :
                                                        inst.op === 'sigmoid' ? SIGMOID_WGSL :
                                                            inst.op === 'tanh' ? TANH_WGSL :
                                                                inst.op === 'sigmoid_backward' ? SIGMOID_BACKWARD_WGSL :
                                                                    inst.op === 'tanh_backward' ? TANH_BACKWARD_WGSL :
                                                                        inst.op === 'cat' ? CAT_WGSL :
                                                                            inst.op === 'where' ? WHERE_WGSL :
                                                                                inst.op === 'dropout' ? DROPOUT_WGSL : '';
                    if (!wgslCode) {
                        throw new AMEVAForgeSecurityError(`Unknown op "${inst.op}"`);
                    }
                    const totalWorkgroups = Math.ceil(numElements / 64);
                    // TS-C01 Fix: 65535 초과 시 2D 그리드로 분산
                    if (totalWorkgroups <= 65535) {
                        dispatchX = totalWorkgroups;
                        dispatchY = 1;
                    }
                    else {
                        // 2D 분산: sqrt로 균등 분할
                        dispatchX = Math.min(65535, Math.ceil(Math.sqrt(totalWorkgroups)));
                        dispatchY = Math.min(65535, Math.ceil(totalWorkgroups / dispatchX));
                    }
                    let numA = 0;
                    let numB = 0;
                    if (inst.in && inst.in.length >= 2) {
                        numA = (idToByteLength[inst.in[0]] ?? byteLength) / 4;
                        numB = (idToByteLength[inst.in[1]] ?? byteLength) / 4;
                    }
                    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, numA, numB, 0, 0, 0, 0]));
                    if (inst.op === 'cat') {
                        if (!inst.params || inst.params.length < 3) {
                            throw new AMEVAForgeSecurityError(`cat instruction missing params`);
                        }
                        const [a_dim, b_dim, stride] = inst.params;
                        // Overwrite the params for cat
                        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([numElements, dispatchX, a_dim, b_dim, stride, 0, 0, 0]));
                    }
                }
                /**
                 * WHAT: 파이프라인 캐시에서 가져온 컴파일된 WebGPU 컴퓨트 파이프라인 객체입니다.
                 * WHY: 커맨드 인코더가 GPU에서 셰이더를 구동하기 위한 명세(Layout)를 설정하기 위해 참조합니다.
                 * HOW: _globalPipelineCache.getPipeline()을 통해 조회 혹은 캐싱 생성하여 얻습니다.
                 */
                const { pipeline } = _globalPipelineCache.getPipeline(inst.op, wgslCode);
                if (inst.op === 'sum' || inst.op === 'max') {
                    if (!inst.in || inst.in.length === 0) {
                        throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
                    }
                    const REDUCTION_WG_SIZE = 256;
                    const reductionInputHandle = idToHandle[inst.in[0]];
                    if (!reductionInputHandle)
                        throw new AMEVAForgeSecurityError(`Unresolved reduction input id ${inst.in[0]}`);
                    let currentByteLength = idToByteLength[inst.in[0]];
                    if (currentByteLength === undefined) {
                        const rec = _globalRegistry.has(reductionInputHandle)
                            ? _globalRegistry.get(reductionInputHandle)
                            : transaction.get(reductionInputHandle);
                        currentByteLength = rec ? rec.byteLength : 4;
                    }
                    let currentSize = currentByteLength / 4;
                    let currentInputBuf = idToBuffer[inst.in[0]];
                    const intermediateAllocations = [];
                    /**
                     * WHAT: 병렬 리덕션(Reduction) 연산을 위한 다중 패스 트리 루프입니다.
                     * WHY: 전체 배열을 하나의 스칼라로 압축하기 위해 여러 번의 컴퓨트 패스를 통해 계층적으로 데이터를 축소시키기 위함입니다.
                     * HOW: 원소 수가 1이 될 때까지 while 루프를 돌며, 임시 버퍼를 만들고 리덕션 패스를 제출하여 크기를 줄여 나갑니다.
                     */
                    while (currentSize > 1) {
                        const numWGs = Math.ceil(currentSize / REDUCTION_WG_SIZE);
                        const { buffer: passBuf, token: passBufToken } = allocateBuffer(Math.max(4, numWGs * 4), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'temporary', `Graph_${instructions[0]?.id}_reduction`);
                        intermediateAllocations.push({ buffer: passBuf, token: passBufToken });
                        const { buffer: passParamsBuf, token: passParamsToken } = allocateBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'uniform', `Graph_${instructions[0]?.id}_reduction_params`);
                        intermediateAllocations.push({ buffer: passParamsBuf, token: passParamsToken });
                        device.queue.writeBuffer(passParamsBuf, 0, new Uint32Array([currentSize, 0, 0, 0]));
                        const wgsl = inst.op === 'sum' ? SUM_WGSL : MAX_WGSL;
                        const { pipeline: reducePipeline } = _globalPipelineCache.getPipeline(inst.op + '_pass', wgsl);
                        const passEncoder = commandEncoder.beginComputePass();
                        passEncoder.setPipeline(reducePipeline);
                        passEncoder.setBindGroup(0, device.createBindGroup({
                            layout: reducePipeline.getBindGroupLayout(0),
                            entries: [
                                { binding: 0, resource: { buffer: passParamsBuf } },
                                { binding: 1, resource: { buffer: currentInputBuf } },
                                { binding: 2, resource: { buffer: passBuf } },
                            ],
                        }));
                        passEncoder.dispatchWorkgroups(numWGs);
                        passEncoder.end();
                        encoderHasCommands = true;
                        currentInputBuf = passBuf;
                        currentSize = numWGs;
                    }
                    commandEncoder.copyBufferToBuffer(currentInputBuf, 0, outBuffer, 0, 4);
                    encoderHasCommands = true;
                    /**
                     * WHAT: 리덕션 연산 중 만들어진 중간 임시 버퍼들을 수집하는 루프입니다.
                     * WHY: 작업 완료 후 가비지 컬렉션이나 명시적 해제를 수행하여 메모리 릭을 방지하기 위함입니다.
                     * HOW: for...of 구문으로 intermediateBuffers 배열을 순회하여 paramsBuffersToDestroy에 등록합니다.
                     */
                    for (const alloc of intermediateAllocations) {
                        paramsAllocations.push(alloc);
                    }
                    continue; // skip normal dispatch
                }
                if (inst.op !== 'fill' && (!inst.in || inst.in.length === 0)) {
                    throw new AMEVAForgeSecurityError(`Instruction op="${inst.op}" is missing 'in' field.`);
                }
                /**
                 * WHAT: 파이프라인 레이아웃에 맞춰 GPUBuffer를 슬롯(binding)에 매핑하는 배열입니다.
                 * WHY: 컴퓨트 셰이더 내부의 @group(0) @binding(N) 변수들과 실제 VRAM 메모리를 연결하기 위해 필요합니다.
                 * HOW: 연산 종류에 따라 분기하여 각 입력 텐서 버퍼들과 출력 버퍼를 순서대로 할당합니다.
                 */
                let bindGroupEntries = [];
                if (inst.op === 'fill') {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: outBuffer } },
                    ];
                }
                else if (inst.op === 'axpy') {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                        { binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } },
                    ];
                }
                else if (inst.op === 'pad') {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                        { binding: 2, resource: { buffer: outBuffer } },
                    ];
                }
                else if (inst.op === 'gather' || inst.op === 'scatter') {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                        { binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } },
                        { binding: 3, resource: { buffer: outBuffer } },
                    ];
                }
                else if (inst.op === 'where') {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                        { binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } },
                        { binding: 3, resource: { buffer: idToBuffer[inst.in[2]] } },
                        { binding: 4, resource: { buffer: outBuffer } },
                    ];
                }
                else if (inst.op === 'dropout') {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                        { binding: 2, resource: { buffer: outBuffer } },
                    ];
                }
                else {
                    bindGroupEntries = [
                        { binding: 0, resource: { buffer: paramsBuffer } },
                        { binding: 1, resource: { buffer: idToBuffer[inst.in[0]] } },
                    ];
                    if (inst.in.length > 1) {
                        bindGroupEntries.push({ binding: 2, resource: { buffer: idToBuffer[inst.in[1]] } });
                        bindGroupEntries.push({ binding: 3, resource: { buffer: outBuffer } });
                    }
                    else {
                        bindGroupEntries.push({ binding: 2, resource: { buffer: outBuffer } });
                    }
                }
                /**
                 * WHAT: 앞서 설정한 bindGroupEntries 리스트를 토대로 생성된 바인드 그룹 객체입니다.
                 * WHY: 실제 컴퓨트 패스 인코더에 setBindGroup을 호출하기 위해 WebGPU의 투명한 핸들로 필요합니다.
                 * HOW: device.createBindGroup을 사용하여 파이프라인 레이아웃 규칙에 맞춰 버퍼들을 확정(commit)합니다.
                 */
                const bindGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: bindGroupEntries
                });
                if (isMatmul) {
                    const MACS_PER_CHUNK = 2_000_000_000;
                    const macsPerRow = N * K;
                    let chunkY = Math.max(1, Math.floor(MACS_PER_CHUNK / macsPerRow));
                    // TS-H01 Fix: Ensure Y dispatch does not exceed 65535 workgroups
                    chunkY = Math.min(chunkY, 65535 * 8);
                    chunkY = Math.min(M, chunkY);
                    /**
                     * WHAT: 행렬 곱셈 연산을 Y축(행) 기준으로 여러 청크(Chunk)로 분할 처리하는 루프입니다.
                     * WHY: 단일 행렬 곱 연산이 너무 거대하여 GPU 실행 한계 시간(Timeout)을 초과하는 TDR 현상을 피하기 위해 작업을 작게 나눕니다.
                     * HOW: for 루프를 통해 offsetY 변수를 증가시키면서 전체 행(M)을 chunkY만큼씩 잘라 컴퓨트 패스를 큐에 넘깁니다.
                     */
                    for (let offsetY = 0; offsetY < M; offsetY += chunkY) {
                        const currentChunkY = Math.min(chunkY, M - offsetY);
                        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([M, N, K, offsetY]));
                        const passEncoder = commandEncoder.beginComputePass();
                        passEncoder.setPipeline(pipeline);
                        passEncoder.setBindGroup(0, bindGroup);
                        passEncoder.dispatchWorkgroups(dispatchX, Math.ceil(currentChunkY / 8), dispatchZ);
                        passEncoder.end();
                        opsInCurrentBatch++;
                        workloadElements += (dispatchX * currentChunkY * 8 * 8);
                        if (offsetY + currentChunkY < M || workloadElements >= WORKLOAD_BUDGET_ELEMENTS || opsInCurrentBatch >= MAX_OPS_PER_SUBMIT) {
                            device.queue.submit([commandEncoder.finish()]);
                            commandEncoder = device.createCommandEncoder();
                            opsInCurrentBatch = 0;
                            workloadElements = 0;
                        }
                    }
                }
                else {
                    const passEncoder = commandEncoder.beginComputePass();
                    passEncoder.setPipeline(pipeline);
                    passEncoder.setBindGroup(0, bindGroup);
                    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
                    passEncoder.end();
                    opsInCurrentBatch++;
                    workloadElements += byteLength / 4;
                    if (workloadElements >= WORKLOAD_BUDGET_ELEMENTS || opsInCurrentBatch >= MAX_OPS_PER_SUBMIT) {
                        device.queue.submit([commandEncoder.finish()]);
                        commandEncoder = device.createCommandEncoder();
                        opsInCurrentBatch = 0;
                        workloadElements = 0;
                    }
                }
            }
        }
        catch (err) {
            // ── 5. Rollback on Sync Error ──
            console.error(`[AMEVA Forge] Transaction Sync Failed. Rolling back...`, err);
            transaction.rollback();
            for (const alloc of paramsAllocations) {
                try {
                    freeBuffer(alloc.buffer, alloc.token);
                }
                catch { }
            }
            // pop scopes to prevent leak
            void device.popErrorScope();
            void device.popErrorScope();
            void device.popErrorScope();
            throw err;
        }
        if (encoderHasCommands || opsInCurrentBatch > 0) {
            device.queue.submit([commandEncoder.finish()]);
            encoderHasCommands = false;
        }
        // ── 5. Commit / Rollback (Async) — await error scopes before returning ──
        const internalError = await device.popErrorScope();
        const oomError = await device.popErrorScope();
        const validationError = await device.popErrorScope();
        // Check for GPU errors BEFORE returning handles
        const gpuError = internalError || oomError || validationError;
        if (gpuError) {
            console.error(`[AMEVA Forge] GPU error detected. Rolling back transaction...`, gpuError);
            transaction.rollback();
            for (const alloc of paramsAllocations) {
                try {
                    freeBuffer(alloc.buffer, alloc.token);
                }
                catch { }
            }
            // Determine error type
            if (internalError) {
                throw new AMEVAForgeInternalGPUError(`Internal GPU Error: ${internalError.message}`);
            }
            else if (oomError) {
                throw new AMEVAForgeOutOfMemoryError(`GPU Out of Memory: ${oomError.message}`);
            }
            else {
                throw new AMEVAForgeValidationError(`GPU Validation Error: ${validationError.message}`);
            }
        }
        // ── 6. Commit transaction to global registry only on verified success ──
        transaction.commit(_globalRegistry);
        // ── 7. Cleanup temporary/uniform allocations after GPU completion ──
        if (paramsAllocations.length > 0) {
            device.queue.onSubmittedWorkDone().then(() => {
                for (const alloc of paramsAllocations) {
                    try {
                        freeBuffer(alloc.buffer, alloc.token);
                    }
                    catch { }
                }
            }).catch(() => {
                for (const alloc of paramsAllocations) {
                    try {
                        freeBuffer(alloc.buffer, alloc.token);
                    }
                    catch { }
                }
            });
        }
        return idToHandle;
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
     *
     * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
     *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
     *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
     */
    /**
     * WHAT: 주어진 객체가 `toJs` 메서드를 가진 PyodideLikeProxy 타입인지 런타임에 확인하는 타입 가드 함수입니다.
     * WHY: 객체의 유효성과 `toJs` 속성의 함수 여부를 동적으로 검사하여, 런타임 에러 없이 안전하게 PyProxy의 데이터를 자바스크립트 영역으로 추출할 수 있도록 보장하는 역할을 합니다.
     * HOW: typeof 연산자와 in 연산자를 사용하여 입력된 값이 객체이며 null이 아니고, 'toJs' 속성이 존재하며 그 타입이 'function'인지 논리식으로 평가하여 불리언 결과를 반환합니다.
     *
     * @param input 검사할 임의의 데이터
     * @returns `toJs` 메서드가 존재하고 함수이면 true
     */
    function hasToJs(input) {
        return (typeof input === "object" &&
            input !== null &&
            "toJs" in input &&
            typeof input.toJs === "function");
    }
    /**
     * WHAT: 주어진 입력 데이터를 검증하고 안전하게 Float32Array 형태로 변환 혹은 반환하는 함수입니다. (H-05 Fix 적용)
     * WHY: 데이터의 중복 복사를 막아(Zero-Copy) 대용량 텐서 데이터 전송 시의 성능 저하를 방지하면서도, 데이터 타입의 일관성을 유지하기 위해 존재합니다.
     * HOW: 이미 Float32Array 형태이면 원본 그대로 반환합니다. PyProxy인 경우 toJs() 결과를 추출한 후 Float32Array이면 그대로 반환하고, ArrayBuffer라면 새로운 Float32Array 뷰로 래핑하여 반환합니다. 그 외에는 예외를 던집니다.
     *
     * @param input Pyodide Proxy 객체이거나 ArrayBuffer/Float32Array 형태의 데이터
     * @returns 확보된 Float32Array
     */
    function ensureFloat32Array(input) {
        /**
         * WHAT: 입력 데이터가 Float32Array 타입인지 확인하는 조건문입니다.
         * WHY: 불필요한 변환 과정을 생략하고 즉시 반환하여 성능(Zero-Copy)을 최적화하기 위함입니다.
         * HOW: instanceof 연산자를 사용하여 입력 객체의 프로토타입 체인을 검사합니다.
         */
        if (input instanceof Float32Array) {
            return input; // H-05: 복사 제거 — 이미 올바른 타입
        }
        /**
         * WHAT: 입력 데이터가 PyodideProxy와 유사한지 확인하는 조건문입니다.
         * WHY: 파이썬으로부터 넘어온 래퍼 객체인 경우 이를 자바스크립트가 인식할 수 있는 원시 버퍼로 추출하기 위해 필요합니다.
         * HOW: 앞서 정의한 hasToJs 타입 가드 함수를 호출하여 조건을 평가합니다.
         */
        if (hasToJs(input)) {
            /**
             * WHAT: 파이썬 객체(PyProxy)로부터 자바스크립트에서 다룰 수 있는 원시 뷰(JS View)나 배열을 추출하여 담는 변수입니다.
             * WHY: 파이썬의 메모리 영역에 있는 데이터를 자바스크립트 타입 시스템 내에서 분석하고 활용하기 위해 존재합니다.
             * HOW: input.toJs() 메서드를 호출하여 반환된 값을 할당합니다.
             */
            const jsView = input.toJs();
            /**
             * WHAT: 추출된 뷰가 Float32Array 타입인지 확인하는 조건문입니다.
             * WHY: WASM 힙 뷰를 이미 올바른 포맷으로 갖고 있다면 또 다른 래핑 없이 바로 재사용하여 메모리 오버헤드를 막기 위함입니다.
             * HOW: instanceof Float32Array 연산자로 타입을 확인한 후 참이면 그대로 리턴합니다.
             */
            if (jsView instanceof Float32Array) {
                return jsView; // H-05: 복사 제거 — WASM 힙 뷰 그대로 반환
            }
            /**
             * WHAT: 추출된 뷰가 ArrayBuffer 타입인지 확인하는 조건문입니다.
             * WHY: 순수 바이트 배열인 경우 우리가 다룰 수 있는 32비트 부동소수점 데이터 뷰(Float32Array)로 해석하기 위해서입니다.
             * HOW: instanceof ArrayBuffer 연산자로 확인한 후, new Float32Array(jsView)를 호출하여 새로운 뷰를 생성하고 리턴합니다.
             */
            if (jsView instanceof ArrayBuffer) {
                return new Float32Array(jsView);
            }
        }
        throw new Error("Invalid input type: expected Float32Array or a Pyodide proxy coercible to Float32Array.");
    }
    /**
     * WHAT: 입력 데이터를 강제로 새로운 메모리 공간에 깊은 복사(Deep Copy)하여 반환하는 함수입니다. (명시적 deep copy 용도)
     * WHY: 원본 데이터(WASM 힙 등)가 삭제되거나 변경되어도 안전하게 데이터를 보존해야 할 때, 혹은 독립적인 데이터 소유권을 가지는 버퍼가 필요할 때 호출하기 위해 존재합니다. 일반 데이터 읽기에는 성능상 사용하지 않아야 합니다.
     * HOW: ensureFloat32Array를 호출하여 먼저 안전한 뷰를 확보한 뒤, new Float32Array(view)를 사용하여 동일한 요소들을 가지는 완전히 새로운 메모리 배열 인스턴스를 할당하여 반환합니다.
     *
     * @param input 원본 데이터
     * @returns 독립된 메모리 공간을 가지는 복사된 Float32Array
     */
    function cloneToFloat32Array(input) {
        /**
         * WHAT: 원본 데이터로부터 읽기 가능한 Float32Array 뷰를 안전하게 가져와 담아두는 변수입니다.
         * WHY: 복사를 수행하기 전, 원본 데이터가 어떤 형태이든(PyProxy, ArrayBuffer 등) 통일된 Float32Array 포맷으로 만들어놓기 위해서입니다.
         * HOW: ensureFloat32Array(input) 함수를 호출하여 반환값을 저장합니다.
         */
        const view = ensureFloat32Array(input);
        return new Float32Array(view);
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *
     * pyodideBridge.ts — globalThis.amevaForge API 등록자
     *
     * H-02 연동: 단일 실행 경로(graphExecutor.ts)로 통합.
     *   executeGraph 시그니처: (instructionsJson: string, jsInputs: unknown) => Record
     *
     * M-06 연동: disposeBatch 추가 (bridge.py의 js_dispose_batch 지원)
     */
    /**
     * WHAT: 여러 개의 텐서 핸들(TensorHandle)을 한 번에 일괄적으로 메모리에서 해제(dispose)합니다.
     * WHY: 파이썬(Pyodide) 환경에서 다수의 텐서 가비지 컬렉션을 효율적으로 처리하기 위해 존재합니다. (단일 호출로 오버헤드 감소)
     * HOW: 반복문(for...of)을 통해 각 핸들마다 GPU 메모리 해제를 시도하며, 이미 해제된 텐서의 에러는 조용히 무시하여 중단되지 않도록 처리합니다.
     *
     * @param handles 해제할 텐서 핸들들의 배열
     */
    function disposeBatch(handles) {
        /**
         * WHAT: 입력받은 핸들 배열을 순회하는 반복문입니다.
         * WHY: 각각의 텐서 리소스에 대해 개별적인 해제 절차가 필요하기 때문에 존재합니다.
         * HOW: for...of 구문을 사용하여 handles 배열의 각 원소(handle)를 하나씩 가져와 내부 블록을 실행합니다.
         */
        for (const handle of handles) {
            /**
             * WHAT: 현재 순회 중인 텐서 핸들이 유효한 값(truthy)인지 확인하는 조건문입니다.
             * WHY: null, undefined 혹은 빈 문자열 같은 잘못된 핸들이 전달되어 불필요한 예외나 시스템 오류가 발생하는 것을 방지하기 위함입니다.
             * HOW: 자바스크립트의 truthy 평가를 통해 handle 값이 존재할 때만 내부의 해제 로직(try-catch 블록)을 수행하도록 제어합니다.
             */
            if (handle) {
                try {
                    dispose(handle);
                }
                catch {
                    /* 이미 해제되었거나 유효하지 않은 경우 예외를 무시하여 전체 일괄 해제 프로세스가 중단되지 않게 합니다. */
                }
            }
        }
    }
    /**
     * WHAT: Pyodide가 자바스크립트 기능에 접근할 수 있도록 전역 `globalThis.amevaForge` 객체를 생성하고 등록합니다.
     * WHY: 파이썬 측 브리지 코드가 WASM을 거쳐 GPU 하드웨어 가속(WebGPU 등) 기능과 그래프 실행 로직을 사용할 수 있게 하는 엔트리 포인트가 필요하기 때문입니다.
     * HOW: 필요한 모든 내부 함수들을 모은 api 객체를 만들고 Object.freeze로 동결시킨 뒤, globalThis의 속성으로 할당하여 전역에서 접근 가능하게 만듭니다.
     *
     * @returns 등록된 전역 API 객체
     */
    function registerPyodideBridge() {
        /**
         * WHAT: 실제로 전역에 노출될 API 객체를 구성하는 변수입니다.
         * WHY: 각 기능(init, read 등)들을 하나의 통일된 인터페이스 객체로 모아서 파이썬 측에서 구조화된 방식으로 쉽게 접근할 수 있게 묶어주기 위함입니다.
         * HOW: AmevaTensorGlobalAPI 타입에 맞추어 내부 모듈에서 임포트한 함수들을 프로퍼티로 할당하여 객체 리터럴을 생성합니다.
         */
        const api = {
            init,
            read,
            dispose,
            getTensorInfo,
            mapBufferAsync,
            readMappedInto,
            executeGraph,
            warmupKernels,
            disposeBatch,
            getQuotaSnapshot,
            snapshotHandles: () => _globalRegistry.snapshotHandles(),
            flushGC: async () => {
                try {
                    const dev = getDevice();
                    await dev.queue.onSubmittedWorkDone();
                }
                catch { }
            },
        };
        Object.freeze(api); // F-014 Fix: API 객체 동결하여 외부 변조 방지
        globalThis.amevaForge = api;
        return api;
    }

    /**
     * AMEVA-Forge Lightweight In-Browser Visual Inspector & DevTools HUD
     * Real-time VRAM allocation tracking & Training loss curve visualization
     */
    let inspectorContainer = null;
    let canvasElement = null;
    let animationFrameId = null;
    const lossHistory = [];
    /**
     * Record a training step loss for live chart visualization
     */
    function recordStepLoss(step, loss) {
        lossHistory.push({ step, loss });
        if (lossHistory.length > 200) {
            lossHistory.shift();
        }
    }
    /**
     * Clear recorded training history
     */
    function clearStepLossHistory() {
        lossHistory.length = 0;
    }
    /**
     * Render HUD loop
     */
    function renderHUD() {
        if (!canvasElement)
            return;
        const ctx = canvasElement.getContext('2d');
        if (!ctx)
            return;
        const width = canvasElement.width;
        const height = canvasElement.height;
        // Background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);
        // Header / Metrics
        const quota = getQuotaSnapshot();
        const handles = _globalRegistry.snapshotHandles();
        ctx.fillStyle = '#f8fafc';
        ctx.font = 'bold 12px monospace';
        ctx.fillText('⚡ AMEVA-Forge DevTools', 10, 20);
        ctx.font = '10px monospace';
        ctx.fillStyle = '#94a3b8';
        const vramKB = (quota.usedBytes / 1024).toFixed(1);
        const maxKB = (quota.maxBytes / (1024 * 1024)).toFixed(0);
        ctx.fillText(`VRAM: ${vramKB} KB / ${maxKB} MB | Handles: ${handles.length}`, 10, 36);
        // VRAM Bar
        const barWidth = width - 20;
        const barHeight = 6;
        ctx.fillStyle = '#334155';
        ctx.fillRect(10, 44, barWidth, barHeight);
        const usageRatio = Math.min(1.0, quota.usedBytes / Math.max(1, quota.maxBytes));
        ctx.fillStyle = usageRatio > 0.8 ? '#ef4444' : '#10b981';
        ctx.fillRect(10, 44, barWidth * usageRatio, barHeight);
        // Loss Curve Area
        const chartX = 10;
        const chartY = 60;
        const chartW = width - 20;
        const chartH = height - 70;
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(chartX, chartY, chartW, chartH);
        if (lossHistory.length > 1) {
            const minLoss = Math.min(...lossHistory.map(h => h.loss));
            const maxLoss = Math.max(...lossHistory.map(h => h.loss), minLoss + 1e-4);
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < lossHistory.length; i++) {
                const x = chartX + (i / (lossHistory.length - 1)) * chartW;
                const normalizedY = (lossHistory[i].loss - minLoss) / (maxLoss - minLoss);
                const y = chartY + chartH - normalizedY * (chartH - 8) - 4;
                if (i === 0) {
                    ctx.moveTo(x, y);
                }
                else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            const latest = lossHistory[lossHistory.length - 1];
            ctx.fillStyle = '#38bdf8';
            ctx.font = '9px monospace';
            ctx.fillText(`Step ${latest.step}: Loss ${latest.loss.toFixed(4)}`, 14, chartY + 12);
        }
        else {
            ctx.fillStyle = '#64748b';
            ctx.font = '9px monospace';
            ctx.fillText('Awaiting training steps...', chartX + 10, chartY + chartH / 2);
        }
        animationFrameId = requestAnimationFrame(renderHUD);
    }
    /**
     * Mount floating DevTools HUD overlay into DOM
     */
    function mountInspector(targetParent) {
        if (inspectorContainer) {
            return inspectorContainer;
        }
        const container = document.createElement('div');
        container.id = 'ameva-forge-devtools';
        container.style.position = 'fixed';
        container.style.bottom = '16px';
        container.style.right = '16px';
        container.style.width = '280px';
        container.style.height = '180px';
        container.style.backgroundColor = '#0f172a';
        container.style.border = '1px solid #334155';
        container.style.borderRadius = '8px';
        container.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.4)';
        container.style.zIndex = '999999';
        container.style.overflow = 'hidden';
        container.style.fontFamily = 'monospace';
        const canvas = document.createElement('canvas');
        canvas.width = 280;
        canvas.height = 180;
        canvas.style.display = 'block';
        container.appendChild(canvas);
        const parent = targetParent || document.body;
        parent.appendChild(container);
        inspectorContainer = container;
        canvasElement = canvas;
        if (typeof requestAnimationFrame !== 'undefined') {
            animationFrameId = requestAnimationFrame(renderHUD);
        }
        return container;
    }
    /**
     * Unmount and destroy DevTools HUD
     */
    function unmountInspector() {
        if (animationFrameId !== null && typeof cancelAnimationFrame !== 'undefined') {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (inspectorContainer && inspectorContainer.parentNode) {
            inspectorContainer.parentNode.removeChild(inspectorContainer);
        }
        inspectorContainer = null;
        canvasElement = null;
    }

    /**
     * Created: 2026-08-12 12:14:52 +0900
     * Modified:
     *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
     *   - 2026-08-17: Security Hardening: Enforce strict public API boundary, isolate raw GPUDevice
     *
     * WHAT: 라이브러리의 외부 공개용(Public) API를 모두 한 곳으로 모아 내보내는 진입점(엔트리포인트) 파일입니다.
     * WHY: 패키지 사용자가 내부 디렉토리 구조를 일일이 알 필요 없이 일관된 단일 경로에서 모듈을 쉽게 임포트할 수 있도록 편의성을 제공하기 위함입니다.
     * HOW: 내부의 여러 모듈들에 정의된 클래스, 타입, 함수 등을 export 및 re-export 키워드를 활용하여 다시 바깥으로 통합 추출합니다.
     */
    /**
     * WHAT: 테스트 환경(E2E / Jest)에서만 제어 가능한 결함 주입(Fault Injection) 훅입니다.
     * WHY: 프로덕션 환경에 raw GPUDevice를 노출하지 않으면서도 OOM, Validation, Device Lost 복구력을 엄격히 검증하기 위함입니다.
     */
    const __testing = ((typeof process !== 'undefined' && process.env && process.env.NODE_ENV === 'test') ||
        (typeof globalThis !== 'undefined' && (globalThis.__AMEVA_TEST_MODE__ || globalThis.testReady))) ? Object.freeze({
        destroyDevice: () => {
            try {
                getDevice().destroy();
            }
            catch { }
            try {
                _resetDeviceForTesting();
            }
            catch { }
        },
        triggerValidationError: async () => {
            const dev = getDevice();
            dev.pushErrorScope('validation');
            try {
                dev.createBuffer({
                    size: 1024,
                    usage: 0, // Usage 0 is an unconditional WebGPU validation fault
                });
            }
            finally {
                const err = await dev.popErrorScope();
                if (err) {
                    throw new AMEVAForgeValidationError(`GPU Validation Error: ${err.message}`);
                }
            }
        },
        setQuotaLimit: (maxBytes) => {
            _globalQuotaManager.setLimits(maxBytes, maxBytes);
        },
        getDeviceInternal: getDevice,
    }) : undefined;

    exports.AMEVAForgeDTypeError = AMEVAForgeDTypeError;
    exports.AMEVAForgeDeviceError = AMEVAForgeDeviceError;
    exports.AMEVAForgeDeviceLostError = AMEVAForgeDeviceLostError;
    exports.AMEVAForgeDisposedError = AMEVAForgeDisposedError;
    exports.AMEVAForgeError = AMEVAForgeError;
    exports.AMEVAForgeInternalGPUError = AMEVAForgeInternalGPUError;
    exports.AMEVAForgeOutOfMemoryError = AMEVAForgeOutOfMemoryError;
    exports.AMEVAForgeQuotaExceededError = AMEVAForgeQuotaExceededError;
    exports.AMEVAForgeSecurityError = AMEVAForgeSecurityError;
    exports.AMEVAForgeShapeError = AMEVAForgeShapeError;
    exports.AMEVAForgeStaleHandleError = AMEVAForgeStaleHandleError;
    exports.AMEVAForgeUnsupportedOpError = AMEVAForgeUnsupportedOpError;
    exports.AMEVAForgeValidationError = AMEVAForgeValidationError;
    exports.AMEVAForgeWebGPUUnavailableError = AMEVAForgeWebGPUUnavailableError;
    exports.KERNEL_REGISTRY = KERNEL_REGISTRY;
    exports.QuotaManager = QuotaManager;
    exports.__testing = __testing;
    exports.add = add;
    exports.assertAllowedKernelName = assertAllowedKernelName;
    exports.assertAllowedShaderConstant = assertAllowedShaderConstant;
    exports.assertSafeShaderIdentifier = assertSafeShaderIdentifier;
    exports.assertStaticShaderSourceOnly = assertStaticShaderSourceOnly;
    exports.assertWasmRange = assertWasmRange;
    exports.clearStepLossHistory = clearStepLossHistory;
    exports.cloneToFloat32Array = cloneToFloat32Array;
    exports.dispose = dispose;
    exports.ensureFloat32Array = ensureFloat32Array;
    exports.executeGraph = executeGraph;
    exports.getAllowedKernelNames = getAllowedKernelNames;
    exports.getQuotaSnapshot = getQuotaSnapshot;
    exports.getTensorInfo = getTensorInfo;
    exports.init = init;
    exports.initWebGPU = initWebGPU;
    exports.isAvailable = isAvailable;
    exports.mapBufferAsync = mapBufferAsync;
    exports.matmul = matmul;
    exports.mountInspector = mountInspector;
    exports.mul = mul;
    exports.random = random;
    exports.read = read;
    exports.readMappedInto = readMappedInto;
    exports.recordStepLoss = recordStepLoss;
    exports.registerKernelNames = registerKernelNames;
    exports.registerPyodideBridge = registerPyodideBridge;
    exports.relu = relu;
    exports.relu_backward = relu_backward;
    exports.resetRuntimeMemory = resetRuntimeMemory;
    exports.transpose = transpose;
    exports.unmountInspector = unmountInspector;
    exports.uploadFloat32Array = uploadFloat32Array;
    exports.validateDType = validateDType;
    exports.validateShape = validateShape;
    exports.warmupKernels = warmupKernels;

}));
//# sourceMappingURL=index.js.map
