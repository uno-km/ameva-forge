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
    except Exception:
        # 해제 중 발생하는 예외는 스레드 안전성이나 기 해제 문제일 수 있으므로 무시합니다.
        pass


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
        # 해제할 핸들 목록이 비어있다면 즉시 반환하여 불필요한 실행을 막습니다.
        return
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 파이썬 객체를 JS 객체로 변환하기 위한 유틸리티를 Pyodide에서 가져옵니다.
    from pyodide.ffi import to_js
    # 파이썬 리스트인 handles를 JS에서 인식할 수 있는 배열로 변환합니다.
    js_handles = to_js(handles)
    
    if hasattr(core, 'disposeBatch'):
        # JS 측에 배치 해제 기능이 존재한다면, 변환된 배열을 넘겨 일괄 해제를 요청합니다.
        core.disposeBatch(js_handles)
    else:
        # JS 측에 배치 기능이 없다면, 반복문을 돌며 기존 단일 해제 메서드를 사용합니다.
        for h in handles:
            # 리스트에 담긴 개별 핸들에 대해 순차적으로 해제를 요청합니다.
            core.dispose(h)


def js_execute_graph(instructions_json: str, inputs: Any) -> Any:
    """
    [WHAT] 
    파이썬 측에서 직렬화한 연산 그래프(명령어)와 입력 데이터들을 JS 측 엔진으로 넘겨 실제 GPU 연산을 실행시키는 핵심 미들웨어 함수입니다.
    
    [WHY] 
    순수 파이썬만으로는 브라우저의 WebGPU API에 접근하여 병렬 연산을 수행할 수 없으므로, 연산 그래프의 직렬화 문자열을 브릿지를 통해 위임해야 합니다.
    
    [HOW] 
    입력 데이터의 dtype을 검증(float32)하고 JS 배열로 변환한 뒤, core.executeGraph()에 명령어와 인풋을 전달하여 실행 결과를 파이썬 딕셔너리로 받아옵니다.
    """
    # 전역 JS 브릿지 객체를 가져옵니다.
    core = get_js_core()
    # 파이썬 객체를 JS 객체로 변환하기 위해 to_js를 임포트합니다.
    from pyodide.ffi import to_js
    # H-NEW-09: inputs 내 ndarray dtype 검증
    # 넘파이 모듈을 불러와 배열 타입을 검사할 준비를 합니다.
    import numpy as np
    
    # 전달받은 입력 데이터 리스트를 순회하며 각 텐서/배열의 요소를 확인합니다.
    for i, inp in enumerate(inputs):
        if hasattr(inp, 'dtype') and inp.dtype != np.float32:
            # 배열의 데이터 타입이 float32가 아니면, 연산 정확도나 GPU 셰이더 호환 문제가 생길 수 있으므로 타입 에러를 던집니다.
            raise TypeError(
                f"Input[{i}] dtype must be float32, got {inp.dtype}. "
                f"Use .astype(np.float32) to convert."
            )
            
    # 입력 리스트를 JS 배열로 변환합니다. depth=1 옵션을 주어 1단계 리스트까지만 변환하도록 제어합니다.
    js_inputs = to_js(inputs, depth=1)

    # 문자열로 된 명령어 JSON과 JS 객체로 변환된 입력 데이터를 전달하여 그래프 연산을 요청합니다.
    result = core.executeGraph(instructions_json, js_inputs)
    
    # js_inputs 프록시 메모리 해제
    if hasattr(js_inputs, 'destroy'):
        js_inputs.destroy()
    # CRITICAL: JS Record<number,string> → Python dict 변환.
    # to_py() 없이는 JS Proxy 객체가 반환되어
    # tensor.py의 out_handles.get() 호출 시 AttributeError 발생.
    if hasattr(result, 'to_py'):
        # JS Proxy 객체일 경우, to_py() 메서드를 통해 순수 파이썬 객체(dict)로 명시적 변환을 수행합니다.
        ret = result.to_py()
        result.destroy() # VUL-019: JS Proxy 객체 해제 (Memory Leak 방지)
        return ret
    # to_py 메서드가 없다면, 기본 dict() 생성자를 사용해 파이썬 딕셔너리로 형변환하여 반환합니다.
    return dict(result)

