"""
errors.py — AMEVA-Tensor 에러 계층 구조

NL-07 Fix: Python errors.py와 TypeScript errors.ts의 에러 클래스 대칭 맞춤.
  추가: AMEVATensorQuotaExceededError, AMEVATensorSecurityError
"""


class AMEVATensorError(Exception):
    """AMEVA-Tensor 기본 에러 클래스."""
    pass


class AMEVATensorShapeError(AMEVATensorError):
    """텐서 shape 불일치 또는 유효하지 않은 shape."""
    pass


class AMEVATensorDTypeError(AMEVATensorError):
    """지원하지 않는 dtype 또는 dtype 불일치."""
    pass


class AMEVATensorDeviceError(AMEVATensorError):
    """디바이스 관련 에러 (기기 불일치, 초기화 실패 등)."""
    pass


class AMEVATensorDisposedError(AMEVATensorError):
    """이미 해제된 텐서에 접근 시 발생."""
    pass


class AMEVATensorWebGPUUnavailableError(AMEVATensorError):
    """WebGPU를 사용할 수 없는 환경 (비-브라우저, 미지원 GPU 등)."""
    pass


class AMEVATensorQuotaExceededError(AMEVATensorError):
    """VRAM 쿼터 초과. NL-07: TypeScript AMEVATensorQuotaExceededError와 대칭."""
    pass


class AMEVATensorSecurityError(AMEVATensorError):
    """보안 위반 (허용되지 않은 op, 셰이더 인젝션 시도 등). NL-07: TypeScript와 대칭."""
    pass
