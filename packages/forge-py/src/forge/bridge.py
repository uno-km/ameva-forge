"""
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
    """현재 런타임이 Pyodide(브라우저 WASM) 환경인지 확인한다."""
    return sys.platform == 'emscripten'


def get_js_core() -> Any:
    """globalThis.amevaForge 바인딩을 반환한다."""
    import js
    if not hasattr(js.globalThis, 'amevaForge'):
        raise RuntimeError(
            "[AMEVA Tensor] WebGPU core not found on globalThis.amevaForge. "
            "Ensure registerPyodideBridge() was called before init()."
        )
    return js.globalThis.amevaForge


def init_bridge(config: Optional[dict] = None) -> Any:
    """JS 단의 WebGPU 엔진을 초기화한다."""
    core = get_js_core()
    return core.init(config)


async def js_map_async(handle: str) -> None:
    core = get_js_core()
    await core.mapBufferAsync(handle)


def js_read_mapped_into(handle: str, arr: Any) -> None:
    core = get_js_core()
    core.readMappedInto(handle, arr)


def js_dispose(handle: str) -> None:
    """단일 핸들의 GPU 버퍼를 해제한다."""
    if handle is None:
        return
    try:
        core = get_js_core()
        core.dispose(handle)
    except Exception:
        pass


def js_dispose_batch(handles: list) -> None:
    """여러 핸들을 한 번의 FFI 호출로 일괄 해제한다 (M-06 Fix)."""
    if not handles:
        return
    core = get_js_core()
    from pyodide.ffi import to_js
    js_handles = to_js(handles)
    if hasattr(core, 'disposeBatch'):
        core.disposeBatch(js_handles)
    else:
        for h in handles:
            core.dispose(h)


def js_execute_graph(instructions_json: str, inputs: Any) -> Any:
    core = get_js_core()
    from pyodide.ffi import to_js
    # H-NEW-09: inputs 내 ndarray dtype 검증
    import numpy as np
    for i, inp in enumerate(inputs):
        if hasattr(inp, 'dtype') and inp.dtype != np.float32:
            raise TypeError(
                f"Input[{i}] dtype must be float32, got {inp.dtype}. "
                f"Use .astype(np.float32) to convert."
            )
    js_inputs = to_js(inputs, depth=1)

    result = core.executeGraph(instructions_json, js_inputs)
    # CRITICAL: JS Record<number,string> → Python dict 변환.
    # to_py() 없이는 JS Proxy 객체가 반환되어
    # tensor.py의 out_handles.get() 호출 시 AttributeError 발생.
    if hasattr(result, 'to_py'):
        return result.to_py()
    return dict(result)

