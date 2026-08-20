"""
errors.py — AMEVA-Forge 에러 계층 구조

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories

NL-07 Fix: Python errors.py와 TypeScript errors.ts의 에러 클래스 대칭 맞춤.
  추가: AMEVAForgeQuotaExceededError, AMEVAForgeSecurityError
"""


class AMEVAForgeError(ValueError):
    """
    AMEVA-Forge 기본 에러 클래스.
    
    무엇을: AMEVA-Forge 패키지 내에서 발생하는 모든 사용자 정의 예외의 최상위 부모 클래스이다.
    왜: 모든 내부 에러를 하나의 예외 타입으로 묶어, 사용자가 AMEVAForgeError 하나만으로 모든 라이브러리 예외를 캐치할 수 있게 하기 위함이다.
    어떻게: 파이썬의 내장 Exception 클래스를 상속받아 구현되었다.
    """
    pass


class AMEVAForgeShapeError(AMEVAForgeError):
    """
    텐서 shape 불일치 또는 유효하지 않은 shape.
    
    무엇을: 텐서 연산 시 요구되는 크기(Shape) 조건이 맞지 않을 때 발생하는 에러 클래스이다.
    왜: 행렬 곱, 덧셈 등 형태(Shape)가 일치해야 하는 연산에서 차원 불일치를 명확히 알리기 위해 존재한다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeDTypeError(AMEVAForgeError):
    """
    지원하지 않는 dtype 또는 dtype 불일치.
    
    무엇을: 텐서의 데이터 타입(dtype)이 지원되지 않거나, 연산 간 데이터 타입이 맞지 않을 때 던지는 에러 클래스이다.
    왜: 호환되지 않는 데이터 타입끼리의 연산을 시도하거나 시스템이 지원하지 않는 타입을 사용할 때 이를 차단하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeDeviceError(AMEVAForgeError):
    """
    디바이스 관련 에러 (기기 불일치, 초기화 실패 등).
    
    무엇을: CPU와 GPU 텐서 간의 연산 등 이기종 디바이스 연산 시도 또는 초기화 문제 발생 시 던지는 에러이다.
    왜: 서로 다른 메모리 공간에 있는 텐서들 간의 연산을 막고, 디바이스 연결 오류를 디버깅하기 쉽게 만들기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeDisposedError(AMEVAForgeError):
    """
    이미 해제된 텐서에 접근 시 발생.
    
    무엇을: 메모리에서 이미 해제(dispose)된 텐서의 자원(데이터나 버퍼)에 다시 접근하려 할 때 발생하는 에러이다.
    왜: 댕글링 포인터 혹은 잘못된 메모리 참조로 인해 시스템이 크래시되는 것을 파이썬 레벨에서 안전하게 방지하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeWebGPUUnavailableError(AMEVAForgeError):
    """
    WebGPU를 사용할 수 없는 환경 (비-브라우저, 미지원 GPU 등).
    
    무엇을: 현재 실행 중인 환경이 WebGPU를 지원하지 않을 때 던져지는 에러 클래스이다.
    왜: Pyodide가 아닌 일반 파이썬 환경이거나 브라우저에서 WebGPU가 활성화되지 않았을 때 유의미한 에러 메시지를 제공하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeQuotaExceededError(AMEVAForgeError):
    """
    VRAM 쿼터 초과. NL-07: TypeScript AMEVAForgeQuotaExceededError와 대칭.
    
    무엇을: 브라우저나 GPU 장치에 할당된 메모리 한도(VRAM Quota)를 초과했을 때 발생하는 에러이다.
    왜: 무분별한 텐서 생성으로 인한 메모리 고갈(OOM) 상황을 명확하게 에러로 감지하기 위해 존재한다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeSecurityError(AMEVAForgeError):
    """
    보안 위반 (허용되지 않은 op, 셰이더 인젝션 시도 등). NL-07: TypeScript와 대칭.
    
    무엇을: 허용되지 않은 보안 정책 위반이나, 악의적인 셰이더 코드 주입 시도 등이 감지되었을 때 발생하는 에러이다.
    왜: 웹 환경(WebGPU)에서 안전하지 않은 동작을 즉각적으로 차단하여 보안 취약점을 예방하기 위함이다.
    어떻게: AMEVAForgeError를 상속받아 세부 예외 타입으로 분리되었다.
    """
    pass


class AMEVAForgeValidationError(AMEVAForgeError):
    """GPU validation error scope에서 감지된 오류."""
    pass


class AMEVAForgeOutOfMemoryError(AMEVAForgeError):
    """GPU out-of-memory error scope에서 감지된 오류."""
    pass


class AMEVAForgeInternalGPUError(AMEVAForgeError):
    """GPU internal error scope에서 감지된 오류."""
    pass


class AMEVAForgeDeviceLostError(AMEVAForgeError):
    """GPU device lost 오류."""
    pass


class AMEVAForgeStaleHandleError(AMEVAForgeError):
    """이전 generation의 stale handle 접근 오류."""
    pass


class AMEVAForgeUnsupportedOperationError(AMEVAForgeDeviceError):
    """Release 1에서 지원하지 않는 연산 오류."""
    pass
