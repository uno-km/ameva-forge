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
export declare class AMEVAForgeError extends Error {
    /**
     * WHAT: AMEVAForgeError 인스턴스를 생성하는 생성자입니다.
     * WHY: 에러 메시지를 초기화하고, 클래스의 인스턴스 타입 체크(instanceof)가 정상적으로 작동하도록 프로토타입 체인을 교정하기 위해 필요합니다.
     * HOW: 부모 생성자(super)를 호출한 후, this.name을 설정하고 Object.setPrototypeOf를 사용하여 프로토타입을 강제로 맞춰줍니다.
     *
     * @param message 사용자에게 노출될 구체적인 에러 메시지 내용
     */
    constructor(message: string);
}
/**
 * WHAT: 텐서의 형태(Shape)나 차원(Dimension)이 계산 또는 검증 중 맞지 않을 때 발생하는 에러 클래스입니다.
 * WHY: 연산의 수학적/구조적 조건이 위배되었음을 사용자나 상위 로직에 명확히 알리기 위해 존재합니다.
 * HOW: AMEVAForgeError를 상속받아 정의되어, Shape 관련된 구체적인 예외 상황을 나타내는 타입으로 활용됩니다.
 */
export declare class AMEVAForgeShapeError extends AMEVAForgeError {
}
/**
 * WHAT: 텐서의 데이터 타입(DType)이 연산에서 지원하지 않거나 서로 충돌할 때 발생하는 에러 클래스입니다.
 * WHY: 잘못된 자료형 접근이나 호환되지 않는 텐서 연산을 조기에 차단하여 런타임 크래시를 방지하기 위해 사용됩니다.
 * HOW: AMEVAForgeError를 상속받아 DType 특화 예외를 표현합니다.
 */
export declare class AMEVAForgeDTypeError extends AMEVAForgeError {
}
/**
 * WHAT: GPU 등 하드웨어 디바이스를 초기화하거나 통신하는 과정에서 발생하는 에러 클래스입니다.
 * WHY: 디바이스 손실(Device Lost)이나 잘못된 디바이스 상태 등 하드웨어 의존적인 실패 상황을 명확히 구분하여 처리하기 위해 필요합니다.
 * HOW: AMEVAForgeError를 상속받아 GPU/디바이스 레벨의 문제를 나타냅니다.
 */
export declare class AMEVAForgeDeviceError extends AMEVAForgeError {
}
/**
 * WHAT: 이미 메모리에서 해제된(disposed) 텐서 자원에 접근하려고 시도할 때 발생하는 에러 클래스입니다.
 * WHY: 메모리 누수나 무효한 메모리 접근(Use-After-Free)을 방지하는 안전장치 역할을 하여, 잘못된 리소스 참조를 차단하기 위함입니다.
 * HOW: AMEVAForgeError를 상속받아 생명주기가 끝난 객체에 대한 접근 시 던져집니다.
 */
export declare class AMEVAForgeDisposedError extends AMEVAForgeError {
}
/**
 * WHAT: 시스템이나 WebGPU에서 할당 가능한 메모리 할당량(Quota)이나 버퍼 크기를 초과했을 때 발생하는 에러 클래스입니다.
 * WHY: 제한된 VRAM이나 시스템 리소스 한계에 도달했음을 명확히 알리고, 메모리 할당 실패를 우아하게(gracefully) 처리하기 위해 존재합니다.
 * HOW: AMEVAForgeError를 상속받아 메모리 관련 한계 초과를 나타냅니다.
 */
export declare class AMEVAForgeQuotaExceededError extends AMEVAForgeError {
}
/**
 * WHAT: 실행 중인 브라우저나 환경이 WebGPU API 자체를 지원하지 않을 때 발생하는 에러 클래스입니다.
 * WHY: 호환되지 않는 환경에서 실행을 시도할 때 발생시켜, 폴백(fallback) 메커니즘을 구동하거나 사용자에게 호환성 문제를 신속히 알리기 위해 사용됩니다.
 * HOW: AMEVAForgeError를 상속받아 WebGPU 초기화 실패 시 즉각적으로 던져집니다.
 */
export declare class AMEVAForgeWebGPUUnavailableError extends AMEVAForgeError {
}
/**
 * WHAT: 보안 정책, 권한 부족, 혹은 검증되지 않은 셰이더/WASM 접근 등 보안 관련된 문제가 발생했을 때 던져지는 에러 클래스입니다.
 * WHY: 비정상적인 메모리 접근이나 권한을 벗어난 조작을 막아 시스템의 전반적인 안전성을 보장하기 위한 보호 계층으로 작용합니다.
 * HOW: AMEVAForgeError를 상속받아 보안 정책 위반 시 발동됩니다.
 */
export declare class AMEVAForgeSecurityError extends AMEVAForgeError {
}
/**
 * WHAT: 현재 구현되지 않았거나 지원하지 않는 연산(Operation)을 실행하려고 할 때 발생하는 에러 클래스입니다.
 * WHY: 사용자가 유효하지 않은 그래프 노드나 현재 라이브러리에서 지원 범위를 벗어난 커널을 호출하는 것을 사전에 막아 오작동을 예방합니다.
 * HOW: AMEVAForgeError를 상속받아 구현되지 않은 기능 호출 시 발생합니다.
 */
export declare class AMEVAForgeUnsupportedOpError extends AMEVAForgeError {
}
