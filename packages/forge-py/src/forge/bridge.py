"""
=============================================================================
[파일 이력 메타데이터]
- Created: 2026-08-12 12:14:52 +0900
- Modified:
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
=============================================================================
bridge.py — Python ↔ TypeScript FFI 미들웨어

NC-01 Fix: is_pyodide() SyntaxError 수정 (try 블록 미완성 → 단순 sys.platform 체크)
NC-02 Fix: Dict 타입힌트 미정의 → Optional[dict] 사용
NC-03 Fix: js_execute_graph()의 *inputs 스프레드 → to_js() 단일 배열 전달
NH-06 Fix: flush_gc() 원자성 — dispose 성공 후에만 큐에서 제거
M-06 Fix: batch dispose 구현
"""
from typing import Any, List, Optional
import sys
import warnings

from .errors import (
    AMEVAForgeValidationError,
    AMEVAForgeOutOfMemoryError,
    AMEVAForgeInternalGPUError,
    AMEVAForgeDeviceLostError,
    AMEVAForgeQuotaExceededError
)


def is_pyodide() -> bool:
    """
    [WHAT] 
    현재 런타임 환경이 Pyodide(웹 어셈블리 브라우저 환경)인지 검사합니다.
    
    [WHY] 
    브라우저 외부(Node.js, CPython)와 내부 환경 간의 동작을 다르게 분기처리하여, FFI 호출 시 에러를 방지하기 위함입니다.
    
    [HOW] 
    파이썬의 sys.platform 값이 'emscripten' 문자열과 일치하는지 비교하여 불리언 결과를 반환합니다.
    """
    # 현재 시스템 플랫폼 이름이 Emscripten 기반인지 확인하여 결과를 반환합니다.
    return sys.platform == 'emscripten'


def is_webgpu_available() -> bool:
    """
    [WHAT] 
    현재 런타임 환경에서 WebGPU FFI 브릿지를 사용할 수 있는지 검사합니다.
    
    [WHY]
    Python 프론트엔드에서 GPU 연산 그래프를 브릿지로 발행하기 전에 WebGPU 런타임 가용성을 확인하기 위함입니다.
    
    [HOW]
    Pyodide 환경 및 CPython FFI 모의 브릿지에서 globalThis.amevaForge 가용성을 검사합니다.
    """
    try:
        import js
        return hasattr(js.globalThis, 'amevaForge')
    except (ImportError, Exception):
        return False


def get_js_core() -> Any:
    """
    [WHAT] 
    자바스크립트 전역 객체인 globalThis.amevaForge 바인딩을 찾아 반환합니다.
    
    [WHY] 
    파이썬 코드에서 자바스크립트로 구현된 WebGPU 엔진의 메서드들에 접근하고 통신할 수 있는 브릿지 객체가 필요하기 때문입니다.
    
    [HOW] 
    Pyodide의 내장 js 모듈을 임포트한 뒤, globalThis에 해당 속성이 존재하는지 검사하고 있으면 객체를 반환, 없으면 RuntimeError를 던집니다.
    """
    # Pyodide에서 자바스크립트 네임스페이스에 접근하기 위해 js 모듈을 임포트합니다.
    import js
    if not hasattr(js.globalThis, 'amevaForge'):
        # JS 전역 공간에 엔진 코어 객체가 등록되어 있지 않다면 치명적인 에러를 발생시켜 실행을 중단합니다.
        raise RuntimeError(
            "[AMEVA Tensor] WebGPU core not found on globalThis.amevaForge. "
            "Ensure registerPyodideBridge() was called before init()."
        )
    # 확인된 JS 엔진 코어 객체를 파이썬 쪽으로 반환합니다.
    return js.globalThis.amevaForge


def init_bridge(config: Optional[dict] = None) -> Any:
    """
    [WHAT] 
    자바스크립트 측 WebGPU 엔진의 초기화를 트리거하는 함수입니다.
    
    [WHY] 
    GPU 디바이스와 렌더링 컨텍스트 등의 필수 인프라를 사용하기 전에 올바르게 셋업되도록 보장해야 하기 때문입니다.
    
    [HOW] 
    get_js_core()로 JS 바인딩을 얻어온 뒤, 전달받은 config 사전을 인자로 넘기며 엔진 코어의 init 메서드를 호출합니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 브릿지 객체의 init 메서드를 호출하여 설정값을 전달하고 실행합니다.
    return core.init(config)


async def js_map_async(handle: str) -> None:
    """
    [WHAT] 
    자바스크립트의 mapBufferAsync 함수를 비동기적으로 호출하는 함수입니다.
    
    [WHY] 
    WebGPU 버퍼 메모리를 파이썬(CPU) 측에서 읽거나 쓸 수 있도록 GPU 메모리를 매핑하는 비동기 처리가 필요하기 때문입니다.
    
    [HOW] 
    get_js_core()를 통해 얻은 JS 코어 객체의 mapBufferAsync 메서드를 await 키워드를 통해 대기(await)하며 실행합니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 주어진 버퍼 핸들에 대해 메모리 매핑을 요청하고 비동기적으로 완료되기를 기다립니다.
    await core.mapBufferAsync(handle)


def js_read_mapped_into(handle: str, arr: Any) -> None:
    """
    [WHAT] 
    매핑된 GPU 버퍼의 데이터를 파이썬 측의 배열(예: TypedArray, 메모리 뷰)로 직접 복사해오는 함수입니다.
    
    [WHY] 
    GPU 연산이 끝난 결과를 파이썬에서 접근 가능한 numpy 배열이나 리스트 형태로 변환하기 위해 데이터를 읽어야 하기 때문입니다.
    
    [HOW] 
    JS 코어의 readMappedInto 메서드에 버퍼 핸들과 데이터를 받을 대상 배열 객체를 넘겨주어 메모리 복사를 수행합니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 매핑된 버퍼의 내용을 대상 객체(arr) 내부로 복사합니다.
    core.readMappedInto(handle, arr)


def js_dispose(handle: str) -> None:
    """
    [WHAT] 
    자바스크립트 측에 할당된 단일 GPU 버퍼 리소스를 해제(dispose)합니다.
    
    [WHY] 
    더 이상 사용되지 않는 텐서 메모리를 방치하면 GPU OOM(Out of Memory)이 발생하므로, 가비지 컬렉터와 연계하여 메모리를 직접 해제해주기 위함입니다.
    
    [HOW] 
    핸들이 유효한지 검사한 후, JS 코어 객체의 dispose 메서드에 핸들을 넘겨 리소스를 삭제하며, 예외 발생 시 무시(pass)합니다.
    """
    if handle is None:
        # 삭제할 핸들이 없으면 아무 작업도 하지 않고 리턴합니다.
        return
    try:
        # 전역 JS 브릿지 객체를 가져옵니다.
        core = get_js_core()
        # 해당 핸들이 가리키는 버퍼 리소스를 해제하도록 요청합니다.
        core.dispose(handle)
    except Exception as e:
        warnings.warn(f"[AMEVA Bridge] GPU handle disposal failed for {handle}: {e}", RuntimeWarning)


def _safe_destroy_proxy(proxy: Any, name: str = "proxy") -> None:
    """
    WHAT: Pyodide FFI 프록시 객체를 안전하게 파괴하고 참조를 무효화하는 표준 헬퍼 함수입니다.
    WHY: 예외를 맹목적으로 무시(pass)하지 않고 진단 정보를 보존하며, 해제 실패 시에도 댕글링 포인터 생성을 막기 위함입니다.
    HOW: hasattr('destroy') 검사 후 파괴를 시도하되, 실패 시 RuntimeWarning을 남깁니다.
    """
    if proxy is None:
        return
    try:
        if hasattr(proxy, 'destroy'):
            proxy.destroy()
    except Exception as e:
        import warnings
        warnings.warn(
            f"[AMEVA-Forge FFI Alert] Failed to destroy Pyodide {name} reference: {e}. "
            f"Reference cleared to prevent use-after-free.",
            RuntimeWarning,
            stacklevel=2,
        )


def js_dispose_batch(handles: list) -> None:
    """
    [WHAT] 
    여러 개의 GPU 버퍼 핸들을 리스트 형태로 받아 한 번의 FFI 호출로 일괄 해제합니다 (M-06 Fix).
    
    [WHY] 
    단일 해제를 반복 호출하는 것은 파이썬과 JS 경계를 넘나드는 FFI 오버헤드가 크기 때문에, 성능을 개선하기 위해 배치 처리를 도입했습니다.
    
    [HOW] 
    핸들 리스트를 JS 배열로 변환(to_js)한 뒤, 코어에 disposeBatch 메서드가 있으면 한 번에 호출하고, 없으면 fallback으로 순회하며 개별 해제합니다.
    """
    if not handles:
        return
    core = get_js_core()
    from pyodide.ffi import to_js
    js_handles = to_js(handles)
    try:
        if hasattr(core, 'disposeBatch'):
            core.disposeBatch(js_handles)
        else:
            for h in handles:
                core.dispose(h)
    finally:
        _safe_destroy_proxy(js_handles, "handles")


def _map_js_error(e: Exception) -> None:
    """Map JS error names to Python typed exceptions with full stack preservation."""
    msg = str(e)
    err_name = getattr(e, 'name', '')
    js_stack = getattr(e, 'stack', '')
    if js_stack and str(js_stack) not in msg:
        msg = f"{msg}\n[JavaScript V8 Stack:\n{js_stack}]"

    from .errors import AMEVAForgeSecurityError

    if 'AMEVAForgeSecurityError' in msg or err_name == 'AMEVAForgeSecurityError':
        raise AMEVAForgeSecurityError(msg) from e
    elif 'AMEVAForgeValidationError' in msg or err_name == 'AMEVAForgeValidationError':
        raise AMEVAForgeValidationError(msg) from e
    elif 'AMEVAForgeOutOfMemoryError' in msg or 'OOM' in msg or err_name == 'AMEVAForgeOutOfMemoryError':
        raise AMEVAForgeOutOfMemoryError(msg) from e
    elif 'AMEVAForgeInternalGPUError' in msg or err_name == 'AMEVAForgeInternalGPUError':
        raise AMEVAForgeInternalGPUError(msg) from e
    elif 'AMEVAForgeDeviceLostError' in msg or 'device lost' in msg.lower() or err_name == 'AMEVAForgeDeviceLostError':
        raise AMEVAForgeDeviceLostError(msg) from e
    elif 'AMEVAForgeQuotaExceededError' in msg or err_name == 'AMEVAForgeQuotaExceededError':
        raise AMEVAForgeQuotaExceededError(msg) from e

async def js_execute_graph(instructions_json: str, inputs) -> dict:
    """Execute graph via JS bridge (async - executeGraph returns Promise)."""
    core = get_js_core()
    js_inputs = None
    result_proxy = None
    try:
        # Convert inputs to JS
        if inputs is not None:
            from pyodide.ffi import to_js
            # Validate float32
            for arr in inputs:
                if hasattr(arr, 'dtype') and str(arr.dtype) != 'float32':
                    raise TypeError(f"Expected float32, got {arr.dtype}")
            js_inputs = to_js(inputs, depth=1)
        
        # Call async executeGraph — returns Promise in Pyodide
        result_proxy = await core.executeGraph(instructions_json, js_inputs)
        
        # Convert JS result to Python dict
        result = result_proxy.to_py()
        return result
    except Exception as e:
        _map_js_error(e)
        raise
    finally:
        if js_inputs is not None:
            if hasattr(js_inputs, 'length'):
                for i in range(len(js_inputs)):
                    try:
                        elem = js_inputs[i]
                        _safe_destroy_proxy(elem, f"input_elem[{i}]")
                    except Exception:
                        pass
            _safe_destroy_proxy(js_inputs, "inputs")
        _safe_destroy_proxy(result_proxy, "result_proxy")

