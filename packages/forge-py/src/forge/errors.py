"""
errors.py — AMEVA-Forge 에러 계층 구조

NL-07 Fix: Python errors.py와 TypeScript errors.ts의 에러 클래스 대칭 맞춤.
  추가: AMEVAForgeQuotaExceededError, AMEVAForgeSecurityError
"""


class AMEVAForgeError(Exception):
    """AMEVA-Forge 기본 에러 클래스."""
    pass


class AMEVAForgeShapeError(AMEVAForgeError):
    """텐서 shape 불일치 또는 유효하지 않은 shape."""
    pass


class AMEVAForgeDTypeError(AMEVAForgeError):
    """지원하지 않는 dtype 또는 dtype 불일치."""
    pass


class AMEVAForgeDeviceError(AMEVAForgeError):
    """디바이스 관련 에러 (기기 불일치, 초기화 실패 등)."""
    pass


class AMEVAForgeDisposedError(AMEVAForgeError):
    """이미 해제된 텐서에 접근 시 발생."""
    pass


class AMEVAForgeWebGPUUnavailableError(AMEVAForgeError):
    """WebGPU를 사용할 수 없는 환경 (비-브라우저, 미지원 GPU 등)."""
    pass


class AMEVAForgeQuotaExceededError(AMEVAForgeError):
    """VRAM 쿼터 초과. NL-07: TypeScript AMEVAForgeQuotaExceededError와 대칭."""
    pass


class AMEVAForgeSecurityError(AMEVAForgeError):
    """보안 위반 (허용되지 않은 op, 셰이더 인젝션 시도 등). NL-07: TypeScript와 대칭."""
    pass
