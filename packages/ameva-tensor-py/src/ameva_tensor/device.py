"""
device.py — WebGPU 디바이스 초기화 및 상태 관리

H-03 Fix: 무성(silent) CPU 폴백 완전 제거.
L-04 Fix: is_pyodide(), init_bridge()를 bridge.py에서 올바르게 임포트.
NM-04 Fix: JS Promise await 방식을 Pyodide 버전 독립적으로 처리.
NM-08 Fix: init() 호출 상태를 추적하여 미초기화 상태에서 GPU 텐서 사용 시 경고.
"""
from typing import Any, Optional
from .bridge import is_pyodide, init_bridge
from .errors import AMEVATensorWebGPUUnavailableError

_CURRENT_DEVICE: str = "cpu"
_INIT_CALLED: bool = False


async def init(experimental_zero_copy: bool = False) -> None:
    """
    WebGPU 엔진을 초기화한다.

    H-03: 초기화 실패 시 CPU 폴백 없이 AMEVATensorWebGPUUnavailableError를 던진다.
    NM-04: Pyodide JS Promise await를 버전 독립적으로 처리한다.

    Raises:
        AMEVATensorWebGPUUnavailableError: Pyodide 환경이 아니거나 WebGPU 초기화 실패 시.
        RuntimeError: JS 브릿지 연결 실패 시.
    """
    global _CURRENT_DEVICE, _INIT_CALLED

    if not is_pyodide():
        raise AMEVATensorWebGPUUnavailableError(
            "AMEVA-Tensor requires a Pyodide (browser/WASM) environment. "
            "Non-browser Python runtimes are not supported for GPU execution."
        )

    # JS 브릿지 초기화 (실패하면 예외 전파 — 폴백 없음)
    res = init_bridge({"experimental_zero_copy": experimental_zero_copy})

    if res is not None:
        import asyncio
        if hasattr(res, 'then') and callable(getattr(res, 'then', None)):
            print("[device.py] JS Promise detected. Awaiting via asyncio.Future bridge...")
            future = asyncio.get_running_loop().create_future()
            def resolve(val): 
                if not future.done(): future.set_result(val)
            def reject(err): 
                if not future.done(): future.set_exception(RuntimeError(str(err)))
            res.then(resolve).catch(reject)
            await future
            print("[device.py] JS Promise resolved successfully.")
        else:
            print(f"[device.py] res is not a Promise. Type: {type(res).__name__}")
            try:
                await res
            except TypeError:
                pass

    _CURRENT_DEVICE = "gpu"
    _INIT_CALLED = True



def is_available() -> bool:
    """현재 WebGPU가 초기화되어 있는지 반환한다."""
    return _CURRENT_DEVICE == "gpu"


def current_device() -> str:
    """현재 기본 디바이스 문자열을 반환한다 ('gpu' 또는 'cpu')."""
    return _CURRENT_DEVICE


def check_gpu_initialized() -> None:
    """
    NM-08 Fix: GPU 텐서 사용 전 초기화 상태를 검사한다.
    초기화되지 않은 상태에서 GPU 텐서를 생성하면 경고를 출력한다.
    """
    if not _INIT_CALLED or _CURRENT_DEVICE != "gpu":
        import warnings
        warnings.warn(
            "GPU device is not initialized. Call 'await at.init()' before creating GPU tensors. "
            "Current device is 'cpu'. GPU operations will fail at realize() time.",
            RuntimeWarning,
            stacklevel=3
        )
