from typing import Any, Tuple, Optional, List
import numpy as np
from .errors import AMEVATensorDisposedError, AMEVATensorShapeError, AMEVATensorDeviceError


def build_lazy_topo(root: 'Tensor') -> List['Tensor']:
    """
    위상 정렬: 레이지 그래프에서 루트까지의 노드를 의존성 순서로 반환.
    Iterative DFS — 재귀 제거 (RecursionError 방지).
    Python 기본 재귀 한도 1000을 초과하는 깊은 모델(Transformer 등)에서도 안전.
    """
    topo: List['Tensor'] = []
    visited: set = set()
    # (node, parent_index) 스택: 현재 노드와 다음에 방문할 부모의 인덱스
    stack: list = [(root, 0)]
    visited.add(id(root))

    while stack:
        node, idx = stack[-1]
        parents = getattr(node, '_parents', ())
        if idx < len(parents):
            # 다음 부모로 진행
            stack[-1] = (node, idx + 1)
            p = parents[idx]
            pid = id(p)
            if pid not in visited:
                visited.add(pid)
                stack.append((p, 0))
        else:
            # 모든 부모 방문 완료 → post-order 추가
            stack.pop()
            topo.append(node)

    return topo


_gc_queue: set = set()


_gc_fail_count: int = 0

def flush_gc() -> None:
    global _gc_fail_count
    if not _gc_queue:
        return
    handles = list(_gc_queue)
    try:
        from .bridge import js_dispose_batch
        js_dispose_batch(handles)
        _gc_queue.difference_update(handles)
        _gc_fail_count = 0
    except Exception:
        _gc_fail_count += 1
        if _gc_fail_count >= 3:
            # 영구 실패: 핸들을 버려서 무한 재시도 방지
            _gc_queue.clear()
            _gc_fail_count = 0


class _HandleCell:
    """
    C-01 Fix: weakref.finalize가 생성 시점의 handle(None)을 캡처하는 버그 방지.
    handle을 mutable container에 담아 finalize 시점에 항상 최신 값을 참조.
    Reference Cell 패턴 (JAX/Linen에서 사용하는 동일 패턴).
    """
    __slots__ = ('handle',)

    def __init__(self, handle: Optional[str]) -> None:
        self.handle = handle


class Tensor:
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: str,
        device: str,
        requires_grad: bool = False,
        handle: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        op: Optional[str] = None,
        parents: tuple = (),
        op_params: Optional[list] = None
    ):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Optional['Tensor'] = None

        # --- 내부 상태 ---
        # C-01: handle을 _HandleCell로 감싸 finalizer가 항상 최신 handle을 참조
        self._handle_cell = _HandleCell(handle)
        self._data = data
        self._disposed = False

        # --- Autograd 상태 ---
        # NC-05 Fix: autograd graph 부모를 별도 필드로 분리
        # _parents: lazy graph traversal용 (레이지 그래프 탐색)
        # _grad_parents: autograd backward용 (Function.apply()에서 설정)
        self._ctx: Optional[Any] = None
        self._parents: tuple = parents
        self._grad_parents: tuple = ()  # NC-05: autograd 전용 부모 필드
        self._op_cls: Optional[Any] = None

        # --- Lazy 그래프 메타데이터 ---
        self._lazy_op = op
        self._lazy_params = op_params

        # RAII: GPU 텐서만 finalize 등록 (C-01: _handle_cell 참조로 수정)
        if self.device == "gpu":
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)

    @property
    def _handle(self) -> Optional[str]:
        return self._handle_cell.handle

    @_handle.setter
    def _handle(self, value: Optional[str]) -> None:
        # C-01: handle 업데이트는 항상 cell을 통해 → finalizer도 최신 값 참조
        self._handle_cell.handle = value

    @staticmethod
    def _finalize_buffer(cell: '_HandleCell') -> None:
        """
        C-01: cell.handle은 realize() 이후 실제 핸들로 채워진 최신 값.
        M-06: 즉시 dispose하지 않고 큐에 모아 Batch GC를 수행한다.
        """
        handle = cell.handle
        if handle is not None:
            _gc_queue.add(handle)

    def _check_disposed(self) -> None:
        if self._disposed:
            raise AMEVATensorDisposedError("Cannot access a disposed Tensor.")

    def realize(self) -> None:
        """레이지 그래프를 단일 FFI 호출로 GPU에 제출한다 (동기 submit, 비동기 실행)."""
        flush_gc()
        if self.device == "cpu" or self._handle is not None:
            return

        topo = build_lazy_topo(self)
        from .graph import GraphBuilder
        builder = GraphBuilder()

        node_id_map: dict = {}
        for v in topo:
            if v._handle is not None:
                # NC-04 Fix: load 노드 — 기존 핸들을 참조, 덮어쓰지 않음
                nid = builder.add_load(v.shape, v._handle)
                node_id_map[id(v)] = nid
            elif v._lazy_op == 'upload':
                nid = builder.add_upload(v.shape, v._data)
                node_id_map[id(v)] = nid
            else:
                in_ids = []
                for p in v._parents:
                    if id(p) not in node_id_map:
                        raise AMEVATensorDeviceError(
                            f"Lazy graph build failed: parent tensor (op={p._lazy_op!r}) "
                            f"is not in the computation graph. It may have been disposed."
                        )
                    in_ids.append(node_id_map[id(p)])
                nid = builder.add_op(v._lazy_op, v.shape, in_ids, v._lazy_params)
                node_id_map[id(v)] = nid

        instructions, inputs = builder.compile()
        from .bridge import js_execute_graph

        out_handles = js_execute_graph(instructions, inputs)

        # NC-04 Fix: load 노드는 이미 _handle이 있으므로 skip,
        # upload/compute 노드만 handle을 업데이트
        for v in topo:
            if v._handle is not None:
                # load 노드: 이미 realized — 덮어쓰지 않음
                continue
            nid = node_id_map[id(v)]
            h = out_handles.get(str(nid)) or out_handles.get(nid)
            if h is None:
                raise AMEVATensorDeviceError(
                    f"Failed to retrieve valid tensor handle from JS for node {nid} "
                    f"(op={v._lazy_op!r}). The JS graph executor may have failed silently."
                )
            v._handle = h  # C-01: _HandleCell을 통해 setter 호출
            if v._lazy_op == 'upload':
                v._data = None  # 호스트 메모리 즉시 해제

    def numpy(self) -> np.ndarray:
        """CPU 텐서의 데이터를 동기적으로 반환한다."""
        self._check_disposed()
        if self.device == "cpu":
            if self._data is None:
                raise AMEVATensorDisposedError("CPU tensor data has been released.")
            return self._data
        else:
            raise AMEVATensorDeviceError(
                "GPU tensor readback is asynchronous. Use: data = await tensor.numpy_async()"
            )

    async def numpy_async(self) -> np.ndarray:
        """GPU 텐서 데이터를 비동기로 읽어온다."""
        self._check_disposed()
        if self.device == "cpu":
            if self._data is None:
                raise AMEVATensorDisposedError("CPU tensor data has been released.")
            return self._data

        # 1. 레이지 그래프를 GPU에 제출 (동기 submit)
        self.realize()

        from .bridge import js_map_async, js_read_mapped_into
        # 2. GPU 큐 완료 대기 + staging 버퍼 맵핑
        await js_map_async(self._handle)

        # 3. WASM 힙에 직접 읽어들이기
        out = np.empty(self.shape, dtype=np.float32)
        js_read_mapped_into(self._handle, out)

        return out

    def dispose(self) -> None:
        """텐서와 연결된 GPU 버퍼를 즉시 해제한다."""
        if self._disposed:
            return
        if self.device == "gpu" and self._handle is not None:
            _gc_queue.add(self._handle)
            self._handle = None


        self._data = None
        self._disposed = True
        self._lazy_op = None
        self._parents = ()
        self._grad_parents = ()  # NC-05: grad parents도 초기화
        self._ctx = None

    def backward(self, gradient: Optional['Tensor'] = None) -> None:
        """역전파를 수행한다."""
        self._check_disposed()
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require grad.")

        from .autograd import build_topological_sort
        from .ops import ones_like

        # NH-09 Fix: 스칼라/비스칼라 구분하여 gradient 초기화
        if gradient is None:
            # PyTorch 방식: 스칼라 or 1-element 텐서만 gradient 없이 backward 가능
            total_elements = 1
            for d in self.shape:
                total_elements *= d
            if total_elements != 1 and self.shape != ():
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            else:
                gradient = ones_like(self)

        # grad 맵 초기화
        grads: dict = {id(self): gradient}

        topo = build_topological_sort(self)
        # 역순 탐색
        for v in reversed(topo):
            if not getattr(v, 'requires_grad', False):
                continue
            if id(v) not in grads:
                continue

            grad_out = grads[id(v)]

            if v._ctx is None or v._op_cls is None:
                # 리프 노드 → grad 누적
                if v.grad is None:
                    v.grad = grad_out
                else:
                    from .ops import add
                    v.grad = add(v.grad, grad_out)
                continue

            # backward 호출 — NC-05 Fix: _grad_parents 사용
            grad_inputs = v._op_cls.backward(v._ctx, grad_out)
            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)

            for parent, g in zip(v._grad_parents, grad_inputs):
                if not getattr(parent, 'requires_grad', False) or g is None:
                    continue
                pid = id(parent)
                if pid in grads:
                    from .ops import add
                    grads[pid] = add(grads[pid], g)
                else:
                    grads[pid] = g



    def relu(self) -> 'Tensor':
        self._check_disposed()
        from .ops import relu
        return relu(self)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        self._check_disposed()
        from .ops import matmul
        return matmul(self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        self._check_disposed()
        from .ops import add
        return add(self, other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        self._check_disposed()
        from .ops import mul
        return mul(self, other)

    def __repr__(self) -> str:
        if self._disposed:
            return '<AMEVA Tensor (disposed)>'
        return (
            f'<AMEVA Tensor shape={self.shape}, dtype={self.dtype}, '
            f'device={self.device}, requires_grad={self.requires_grad}, '
            f'handle={self._handle}>'
        )
