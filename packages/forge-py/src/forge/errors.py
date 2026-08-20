"""
errors.py — AMEVA-Forge 에러 계층 구조

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories

NL-07 Fix: Python errors.py와 TypeScript errors.ts의 에러 클래스 대칭 맞춤.
  추가: AMEVAForgeQuotaExceededError, AMEVAForgeSecurityError
"""


class AMEVAForgeError(Exception):
    """AMEVA-Forge base error class."""
    pass


class AMEVAForgeShapeError(AMEVAForgeError, ValueError):
    """Tensor shape mismatch or invalid shape."""
    pass


class AMEVAForgeDTypeError(AMEVAForgeError, TypeError):
    """Unsupported dtype or dtype mismatch."""
    pass


class AMEVAForgeDeviceError(AMEVAForgeError, RuntimeError):
    """Device-related error (device mismatch, initialization failure)."""
    pass


class AMEVAForgeDisposedError(AMEVAForgeError, RuntimeError):
    """Accessing already disposed tensor resources."""
    pass


class AMEVAForgeWebGPUUnavailableError(AMEVAForgeDeviceError):
    """WebGPU not available in current environment."""
    pass


class AMEVAForgeQuotaExceededError(AMEVAForgeError, RuntimeError):
    """VRAM Quota exceeded."""
    pass


class AMEVAForgeSecurityError(AMEVAForgeError, ValueError):
    """Security violation (prohibited op, shader injection, pickle payload)."""
    pass


class AMEVAForgeValidationError(AMEVAForgeError, ValueError):
    """GPU validation error scope or invalid parameter error."""
    pass


class AMEVAForgeOutOfMemoryError(AMEVAForgeError, RuntimeError):
    """GPU out-of-memory error scope."""
    pass


class AMEVAForgeInternalGPUError(AMEVAForgeError, RuntimeError):
    """GPU internal error scope."""
    pass


class AMEVAForgeDeviceLostError(AMEVAForgeError, RuntimeError):
    """GPU device lost error."""
    pass


class AMEVAForgeStaleHandleError(AMEVAForgeError, RuntimeError):
    """Accessing stale handle from previous generation."""
    pass


class AMEVAForgeUnsupportedOperationError(AMEVAForgeDeviceError):
    """Unsupported operation error."""
    pass
