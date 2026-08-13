"""
device.py — WebGPU 디바이스 초기화 및 상태 관리

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories

H-03 Fix: 무성(silent) CPU 폴백 완전 제거.
L-04 Fix: is_pyodide(), init_bridge()를 bridge.py에서 올바르게 임포트.
NM-04 Fix: JS Promise await 방식을 Pyodide 버전 독립적으로 처리.
NM-08 Fix: init() 호출 상태를 추적하여 미초기화 상태에서 GPU 텐서 사용 시 경고.
"""
from typing import Any, Optional
from .bridge import is_pyodide, init_bridge
from .errors import AMEVAForgeWebGPUUnavailableError

# 현재 사용 중인 디바이스 상태를 저장하는 전역 변수 ('cpu' 또는 'gpu')
# 무엇을: 활성화된 디바이스 상태를 문자열로 보관한다.
# 왜: 현재 시스템이 CPU를 쓰는지 GPU를 쓰는지 전역적으로 추적하기 위해 존재한다.
# 어떻게: 초기화 전에는 'cpu'를 가지며, WebGPU가 성공적으로 로드된 후 'gpu'로 변경된다.
_CURRENT_DEVICE: str = "cpu"

# WebGPU 엔진 초기화 함수(init)가 호출되었는지 여부를 저장하는 전역 변수
# 무엇을: 시스템 초기화 완료 여부를 불리언 값으로 보관한다.
# 왜: 초기화되지 않은 상태에서 GPU 자원을 요청할 때 적절한 경고를 주기 위해 존재한다.
# 어떻게: init() 함수가 성공적으로 완료된 후 True로 상태가 변경된다.
_INIT_CALLED: bool = False


async def init(experimental_zero_copy: bool = False) -> None:
    """
    WebGPU 엔진을 초기화한다.

    무엇을: 비동기 방식으로 브라우저의 WebGPU 기능을 활성화하고 런타임 환경을 셋업하는 역할을 한다.
    왜: WebGPU 기반의 텐서 연산을 수행하기 위해 반드시 브라우저 환경 및 JS 브릿지가 초기화되어야 하기 때문이다.
    어떻게: Pyodide 환경인지 검증한 뒤 JS 브릿지를 로드하며, 반환된 JS Promise를 Python의 asyncio Future로 매핑하여 비동기 대기를 수행한다.

    H-03: 초기화 실패 시 CPU 폴백 없이 AMEVAForgeWebGPUUnavailableError를 던진다.
    NM-04: Pyodide JS Promise await를 버전 독립적으로 처리한다.

    Raises:
        AMEVAForgeWebGPUUnavailableError: Pyodide 환경이 아니거나 WebGPU 초기화 실패 시.
        RuntimeError: JS 브릿지 연결 실패 시.
    """
    global _CURRENT_DEVICE, _INIT_CALLED

    if not is_pyodide():
        raise AMEVAForgeWebGPUUnavailableError(
            "AMEVA-Forge requires a Pyodide (browser/WASM) environment. "
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
                # 무엇을: JS Promise가 성공적으로 완료(resolve)되었을 때 호출될 콜백 함수이다.
                # 왜: JS에서 완료된 결과를 Python의 asyncio.Future에 성공 상태로 반영하기 위해 존재한다.
                # 어떻게: Future 객체가 아직 완료되지 않았다면 set_result 메서드를 통해 결과 값을 설정한다.
                if not future.done(): future.set_result(val)
            def reject(err): 
                # 무엇을: JS Promise가 실패(reject)되었을 때 호출될 에러 처리 콜백 함수이다.
                # 왜: JS 측에서 발생한 에러를 Python 레벨의 예외로 전환 및 전파하기 위해 존재한다.
                # 어떻게: Future 객체가 아직 완료되지 않았다면 RuntimeError와 함께 set_exception을 호출하여 예외를 발생시킨다.
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
    """
    현재 WebGPU가 초기화되어 있는지 반환한다.
    
    무엇을: 현재 실행 환경에서 WebGPU 사용이 가능한 상태인지 여부를 불리언(Boolean) 값으로 반환한다.
    왜: 텐서 연산 시 사용 가능한 디바이스(WebGPU 지원 여부)에 따라 동적인 분기를 처리하기 위해 필요하다.
    어떻게: _CURRENT_DEVICE 전역 변수의 값이 "gpu"인지 직접 비교 연산하여 그 결과를 반환한다.
    """
    return _CURRENT_DEVICE == "gpu"


def current_device() -> str:
    """
    현재 기본 디바이스 문자열을 반환한다 ('gpu' 또는 'cpu').
    
    무엇을: 현재 시스템에 활성화된 연산 디바이스의 이름을 문자열 형태로 반환한다.
    왜: 텐서 생성 시 디폴트 디바이스 값을 결정하거나 현재 활성 디바이스 상태를 외부 모듈에서 조회할 수 있도록 하기 위해 존재한다.
    어떻게: _CURRENT_DEVICE 전역 변수에 저장된 문자열 값을 단순히 리턴하여 제공한다.
    """
    return _CURRENT_DEVICE


def check_gpu_initialized() -> None:
    """
    NM-08 Fix: GPU 텐서 사용 전 초기화 상태를 검사한다.
    초기화되지 않은 상태에서 GPU 텐서를 생성하면 경고를 출력한다.
    
    무엇을: GPU 자원을 사용하려는 시점에서 엔진이 올바르게 초기화(init)되었는지 확인하고, 그렇지 않다면 런타임 경고 메시지를 발생시킨다.
    왜: 사용자가 실수로 초기화 함수를 호출하지 않고 GPU 텐서를 할당하려 할 때, 나중에 연산 시점에서 발생할 수 있는 크래시에 대해 선제적으로 안내하기 위함이다.
    어떻게: 전역 상태인 _INIT_CALLED와 _CURRENT_DEVICE를 조건문으로 검사하고, 초기화 요건을 충족하지 못했다면 warnings 모듈을 통해 적절한 경고 메시지를 띄운다.
    """
    if not _INIT_CALLED or _CURRENT_DEVICE != "gpu":
        import warnings
        warnings.warn(
            "GPU device is not initialized. Call 'await at.init()' before creating GPU tensors. "
            "Current device is 'cpu'. GPU operations will fail at realize() time.",
            RuntimeWarning,
            stacklevel=3
        )
