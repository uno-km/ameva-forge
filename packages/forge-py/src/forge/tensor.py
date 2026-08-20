"""
================================================================================
파일 이력 (Historical Metadata)
Created: 2026-08-12 12:14:52 +0900 (첫 커밋 기준)
Modified:
  - 2026-08-12 12:23:09 +0900: Docs: Build Apache-style docs and unify tests
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
================================================================================
이 파일은 핵심 텐서 자료구조와 지연 평가(Lazy Evaluation), 자동 미분(Autograd)을 위한 그래프 구성 기능을 구현합니다.
"""
# 타입 힌트를 위한 타이핑 모듈 임포트
from typing import Any, Tuple, Optional, List
# 넘파이 배열 조작을 위한 임포트
import numpy as np
# 패키지 내부 커스텀 에러 클래스 임포트
from .errors import AMEVAForgeDisposedError, AMEVAForgeShapeError, AMEVAForgeDeviceError


def build_lazy_topo(root: 'Tensor') -> List['Tensor']:
    """
    WHAT: 레이지 텐서 연산 그래프의 위상 정렬(Topological Sort)을 수행하여 루트부터 리프 노드까지의 순서를 반환하는 함수입니다.
    WHY: 연산을 실행(Realize)할 때 의존성이 있는 부모 텐서들이 먼저 계산되어야 하므로 올바른 실행 순서를 보장하기 위함입니다.
    HOW: 재귀 호출로 인한 스택 오버플로우를 막기 위해 반복적 깊이 우선 탐색(Iterative DFS) 방식으로 구현되었습니다.
    """
    # WHAT: 최종 위상 정렬 결과를 담을 텐서 리스트입니다.
    # WHY: 부모 노드 방문이 모두 완료된 후(Post-order) 텐서를 순차적으로 담아 반환하기 위해서입니다.
    # HOW: 노드의 부모들을 모두 방문하고 나면 리스트의 끝에(append) 추가합니다.
    topo: List['Tensor'] = []
    
    # WHAT: 이미 방문한 노드의 메모리 주소(ID)를 기록하는 집합(Set)입니다.
    # WHY: 그래프 내에서 동일한 노드를 중복으로 방문하거나 무한 루프에 빠지는 것을 방지하기 위함입니다.
    # HOW: 각 노드를 스택에 넣을 때 그 ID를 이 집합에 추가하고 확인합니다.
    visited: set = set()
    
    # (node, parent_index) 스택: 현재 노드와 다음에 방문할 부모의 인덱스
    # WHAT: 순회를 제어하기 위한 명시적인 작업 스택입니다.
    # WHY: 시스템 재귀 호출 한도 초과 문제를 우회하고 더 깊은 모델도 처리할 수 있게 하기 위해 반복문 기반 스택을 씁니다.
    # HOW: 시작 노드(root)와 첫 부모 인덱스(0)를 튜플로 묶어 리스트에 넣습니다.
    stack: list = [(root, 0)]
    visited.add(id(root))

    # WHAT: 스택에 남아있는 작업이 없을 때까지 반복하는 메인 루프입니다.
    # WHY: 그래프에 연결된 모든 노드를 순회하여 빠짐없이 정렬된 리스트를 만들기 위해서입니다.
    # HOW: 스택이 비어있지 않으면 계속 루프를 실행합니다.
    while stack:
        # WHAT: 스택의 가장 위(마지막)에 있는 현재 텐서 노드와 방문할 부모 인덱스를 가져옵니다.
        # WHY: DFS 방식이므로 가장 최근에 추가된(깊이가 깊은) 노드부터 처리하기 위함입니다.
        # HOW: 스택 리스트의 마지막 요소([-1])를 튜플 언패킹합니다.
        node, idx = stack[-1]
        
        # WHAT: 현재 텐서의 부모 노드 튜플입니다.
        # WHY: 이미 GPU 버퍼가 실체화(realize)된 텐서는 리프(load) 노드로 취급하여 부모를 재탐색하지 않음으로써 그래프 낭비와 고아 텐서 참조를 방지합니다.
        # HOW: _handle이 None이 아닐 경우 빈 튜플로 처리합니다.
        if getattr(node, '_handle', None) is not None:
            parents = ()
        else:
            parents = getattr(node, '_parents', ())

        # WHAT: 아직 방문하지 않은 부모 노드가 남아있는지 검사하는 조건문입니다.
        # WHY: 모든 부모 노드를 먼저 처리한 뒤에야 현재 노드를 처리(위상 정렬 리스트 추가)할 수 있기 때문입니다.
        # HOW: 현재 인덱스(idx)가 전체 부모 개수보다 작은지 비교합니다.
        if idx < len(parents):
            stack[-1] = (node, idx + 1)
            p = parents[idx]
            if p is not None and id(p) not in visited:
                visited.add(id(p))
                stack.append((p, 0))
        else:
            stack.pop()
            topo.append(node)

    return topo


# WHAT: GPU 리소스 해제가 필요한 텐서 핸들(문자열 등)을 임시로 모아두는 큐(집합)입니다.
# WHY: 단일 텐서마다 즉각적으로 리소스를 해제(dispose)하면 오버헤드가 크므로, 모아서 일괄 처리(Batch GC)하기 위함입니다.
# HOW: Python의 set 객체를 전역으로 생성해 중복 핸들 등록을 방지하고 최대 크기를 제한합니다.
_gc_queue: set = set()
_gc_queued_bytes: int = 0
_GC_BYTE_THRESHOLD: int = 32 * 1024 * 1024  # 32 MB
_MAX_GC_QUEUE_SIZE: int = 10_000  # 최대 고아 핸들 큐 크기 제한 (메모리 고갈 방어)

_gc_failures: int = 0
_gc_next_retry_at: float = 0.0

def flush_gc(force: bool = False) -> None:
    """
    WHAT: 보류 중인(큐에 쌓인) 리소스 해제 요청들을 모아 JS/WebGPU 브릿지로 일괄 전달하여 처리하는 함수입니다.
    WHY: 성능 최적화를 위해 개별 해제 대신 Batch Dispose를 수행하며, 브릿지 일시 지연 시 지수 백오프로 재시도하고,
         연속 5회 이상 실패 시 메모리 누수를 막기 위해 큐를 강제 소각(Purge)합니다.
    HOW: 큐에 항목이 있으면 백오프 타임을 체크한 후 js_dispose_batch를 호출하고, 배치 성공 시에만 큐에서 제거합니다.
    """
    global _gc_failures, _gc_next_retry_at, _gc_queued_bytes
    import time
    import warnings
    
    if not _gc_queue:
        _gc_queued_bytes = 0
        return
        
    now = time.monotonic()
    if not force and now < _gc_next_retry_at:
        return
        
    handles = list(_gc_queue)
    try:
        from .bridge import js_dispose_batch
        js_dispose_batch(handles)
        _gc_queue.difference_update(handles)
        _gc_queued_bytes = 0
        _gc_failures = 0
        _gc_next_retry_at = 0.0
    except Exception as e:
        _gc_failures += 1
        if _gc_failures >= 10:
            # WHAT: WebGPU 브릿지 응답 불가 시 고아 핸들 큐를 강제 소각합니다.
            # WHY: 호스트 CPU 메모리가 고갈(OOM DoS)되는 것을 차단하기 위함입니다.
            warnings.warn(
                f"[AMEVA-Forge Resource Alert] WebGPU bridge unresponsive for 10 consecutive attempts. "
                f"Purging {len(_gc_queue)} stale GPU handles to prevent host memory leak. (Subsequent WebGPU calls may be degraded): {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            _gc_queue.clear()
            _gc_queued_bytes = 0
            _gc_failures = 0
            _gc_next_retry_at = 0.0
        else:
            delay = min(2.0 ** _gc_failures, 30.0)
            _gc_next_retry_at = now + delay
            warnings.warn(
                f"[AMEVA-Forge GC Warning] disposeBatch failed ({_gc_failures}/10); keeping {len(_gc_queue)} handles queued for retry in {delay:.1f}s: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


class _HandleCell:
    """
    WHAT: 실제 핸들과 할당 바이트 크기를 감싸고 참조 카운트(Ref Count)를 관리하는 레퍼런스 셀 클래스입니다.
    WHY: PyTorch c10::StorageImpl 표준을 채택하여, Tensor.detach()나 View 텐서들이 동일한 GPU 버퍼를 안전하게 공유하고,
         모든 텐서 참조가 완전히 소멸될 때만 단 1회 WebGPU VRAM 버퍼를 해제하여 Use-After-Free와 Double Free를 원천 차단하기 위함입니다.
    HOW: ref_count 정수를 관리하며, inc_ref()와 dec_ref()를 통해 수명주기를 추적합니다.
    """
    __slots__ = ('handle', 'byte_length', 'ref_count')

    def __init__(self, handle: Optional[str], byte_length: int = 0) -> None:
        """
        WHAT: HandleCell 객체의 생성자입니다.
        WHY: 객체 생성 시 초기 핸들 값, 예상 바이트 크기, 초기 참조 카운트(1)를 설정하기 위함입니다.
        HOW: 전달받은 인자들을 멤버 변수에 할당합니다.
        """
        self.handle = handle
        self.byte_length = byte_length
        self.ref_count = 1

    def inc_ref(self) -> None:
        """참조 카운트를 1 증가시킵니다."""
        self.ref_count += 1

    def dec_ref(self) -> bool:
        """참조 카운트를 1 감소시키며, 마지막 참조가 사라졌는지(<= 0) 반환합니다."""
        self.ref_count -= 1
        return self.ref_count <= 0


class Tensor:
    """
    WHAT: AMEVA-Forge의 핵심 데이터 구조인 텐서 클래스입니다.
    WHY: 다차원 배열 데이터를 다루고, 연산 기록을 추적하여 자동 미분 및 GPU 레이지 평가를 지원하기 위함입니다.
    HOW: 내부적으로 shape, dtype, device 등의 메타데이터를 저장하며, 데이터 또는 연산(AST)의 참조를 유지합니다.
    """
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
        op_params: Optional[list] = None,
        handle_cell: Optional['_HandleCell'] = None,
    ):
        """
        WHAT: 텐서 객체를 초기화하는 생성자입니다.
        WHY: 텐서의 형태, 타입, 디바이스 위치 및 연산 히스토리를 설정하기 위함입니다.
        HOW: 전달받은 인자들을 검증하고 멤버 변수들에 할당합니다.
        """
        # WHAT: 텐서의 차원 크기를 나타내는 튜플입니다.
        # WHY: 각 연산에서 형태(Shape) 검증 및 브로드캐스팅에 사용됩니다.
        # HOW: 입력받은 shape 튜플을 할당합니다.
        self.shape = shape

        # PY-H01 Fix: shape 타입 검증
        if not isinstance(shape, tuple):
            raise AMEVAForgeShapeError(f"shape must be a tuple, got {type(shape).__name__}")
        
        # WHAT: shape 튜플의 각 차원(d)을 순회하는 반복문입니다.
        # WHY: 모든 차원이 유효한 양의 정수인지 검증하기 위함입니다.
        # HOW: enumerate로 인덱스와 값을 가져와 int 타입 및 0 이상인지 체크합니다.
        for i, d in enumerate(shape):
            if not isinstance(d, int):
                raise AMEVAForgeShapeError(f"shape[{i}] must be int, got {type(d).__name__}: {d}")
            if d < 0:
                raise AMEVAForgeShapeError(f"shape[{i}] must be non-negative, got {d}")

        # PY-H02 Fix: 빈 텐서(0-element) 차단 — GPU에서 0바이트 버퍼 크래시 방지
        # WHAT: shape 내에 0이 포함되어 있는지 확인하는 제너레이터(암묵적 반복문)입니다.
        # WHY: WebGPU 등에서 0 크기의 버퍼 할당 시 크래시가 발생할 수 있기 때문입니다.
        # HOW: any()와 제너레이터 표현식을 사용하여 0이 있는지 검사합니다.
        if any(d == 0 for d in shape) and len(shape) > 0:
            raise AMEVAForgeShapeError(
                f"Zero-size dimensions are not supported: shape={shape}. "
                f"All dimensions must be positive."
            )

        # Rank 제한: TS MAX_SHAPE_DIM=8과 일치
        if len(shape) > 8:
            raise AMEVAForgeShapeError(
                f"Maximum tensor rank is 8, got {len(shape)}."
            )

        # WHAT: 텐서 데이터의 자료형(예: 'float32')입니다.
        # WHY: 메모리 할당 크기 및 통신 시 타입 일치를 위해 필요합니다.
        # HOW: 인자로 받은 dtype을 할당합니다.
        self.dtype = dtype
        
        # WHAT: 텐서가 저장될 장치('cpu' 또는 'gpu')입니다.
        # WHY: 데이터 저장소 및 연산 수행 장소를 결정하기 위함입니다.
        # HOW: 인자로 받은 device를 할당합니다.
        self.device = device
        
        # WHAT: 정수형/불리언 등 비미분 Dtype의 requires_grad=True 설정 차단 (PyTorch 불변식 준수)
        # WHY: 이산 데이터에는 기울기가 정의되지 않으며, 잘못된 Autograd 노드 생성을 조기에 차단하기 위함입니다.
        if requires_grad and str(dtype) in ("int8", "int16", "int32", "int64", "uint8", "bool"):
            from .errors import AMEVAForgeValidationError
            raise AMEVAForgeValidationError(
                f"[AMEVA-Forge Autograd Error] Only Tensors of floating point dtype can require gradients, got dtype='{dtype}'"
            )
        self.requires_grad = requires_grad
        
        # WHAT: 역전파를 통해 계산된 이 텐서의 기울기(Gradient)입니다.
        # WHY: 파라미터 업데이트를 위해 기울기를 저장해두어야 하기 때문입니다.
        # HOW: 초기값 None으로 설정됩니다.
        self.grad: Optional['Tensor'] = None

        # --- 내부 상태 ---
        # WHAT: 실제 텐서 핸들(GPU 등)을 간접 참조하기 위한 래퍼 객체입니다.
        # WHY: PyTorch c10::StorageImpl 패턴: handle을 _HandleCell로 감싸고 ref_count를 관리하여 안전한 View/Detach를 지원합니다.
        # HOW: 전달받은 handle_cell이 있으면 재사용하고(inc_ref), 없으면 새로 생성합니다.
        if handle_cell is not None:
            self._handle_cell = handle_cell
            if self.device == "gpu":
                self._handle_cell.inc_ref()
        else:
            elem_count = 1
            for d in shape:
                elem_count *= max(d, 1)
            dtype_bytes = 2 if dtype in ('float16', 'int16') else 4
            byte_length = elem_count * dtype_bytes
            self._handle_cell = _HandleCell(handle, byte_length)
        
        # WHAT: CPU에 저장된 텐서의 실제 데이터(numpy 배열)입니다.
        # WHY: CPU 기반 연산이나 읽어온 데이터를 캐싱하기 위함입니다.
        # HOW: 인자로 받은 data를 할당합니다.
        self._data = data
        
        # WHAT: 텐서 리소스가 명시적으로 해제되었는지 여부 플래그입니다.
        # WHY: 이미 해제된 텐서에 접근하는 것을 막아 안전성을 확보하기 위함입니다.
        # HOW: False로 초기화합니다.
        self._disposed = False
        self._version = 0

        # --- Autograd 상태 ---
        # WHAT: 역전파 시 활용될 컨텍스트 객체입니다.
        # WHY: 순전파(forward) 시에 역전파에 필요한 임시 값들을 저장하기 위함입니다.
        # HOW: None으로 초기화합니다.
        self._ctx: Optional[Any] = None
        
        # WHAT: 이 텐서를 만들어낸 부모 텐서들(레이지 그래프 탐색용)입니다.
        # WHY: 레이지 그래프를 위상 정렬로 탐색하기 위함입니다.
        # HOW: 인자로 받은 parents 튜플을 할당합니다.
        self._parents: tuple = parents
        
        # WHAT: 자동 미분(backward) 전용 부모 텐서 튜플입니다.
        # WHY: NC-05: autograd 그래프와 레이지 AST 그래프의 의존성을 분리하기 위함입니다.
        # HOW: 빈 튜플로 초기화합니다.
        self._grad_parents: tuple = ()
        
        # WHAT: 이 텐서를 생성한 연산(Operation)의 클래스 참조입니다.
        # WHY: 역전파 시 해당 클래스의 backward를 호출하기 위함입니다.
        # HOW: None으로 초기화합니다.
        self._op_cls: Optional[Any] = None

        # --- Lazy 그래프 메타데이터 ---
        # WHAT: 레이지 평가 시 이 텐서를 생성하기 위한 연산 이름입니다.
        # WHY: GPU 컴파일러가 어떤 연산을 수행해야 할지 알기 위함입니다.
        # HOW: 인자로 받은 op 문자열을 할당합니다.
        self._lazy_op = op
        
        # WHAT: 레이지 연산에 필요한 추가 파라미터 리스트입니다.
        # WHY: 차원(axis) 축 등 연산별 고유 옵션을 저장하기 위함입니다.
        # HOW: 인자로 받은 op_params를 할당합니다.
        self._lazy_params = op_params

        # WHAT: GPU 장치인 경우 텐서 파괴 시 가비지 컬렉션을 수행하는 코드 블록입니다.
        # WHY: 파이썬 객체 소멸 시 메모리 누수를 방지하고 GPU 리소스도 해제하기 위함입니다.
        # HOW: weakref.finalize를 사용해 콜백을 등록합니다.
        self._finalizer_registered = False
        if self.device == "gpu":
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
            self._finalizer_registered = True
            if len(_gc_queue) >= 16 or _gc_queued_bytes >= _GC_BYTE_THRESHOLD:
                flush_gc()

    @property
    def _handle(self) -> Optional[str]:
        # WHAT: 내부 래퍼(HandleCell)에서 실제 핸들 값을 가져옵니다.
        # WHY: 텐서의 고유 ID(GPU 상의 버퍼 포인터 등)를 확인하기 위함입니다.
        # HOW: _handle_cell의 handle 속성을 반환합니다.
        return self._handle_cell.handle

    @_handle.setter
    def _handle(self, value: Optional[str]) -> None:
        # WHAT: 내부 래퍼에 새로운 핸들 값을 설정합니다.
        # WHY: 연산 결과로 새로운 버퍼가 할당되었을 때 참조를 갱신하기 위함입니다.
        # HOW: _handle_cell의 handle 속성에 값을 대입합니다.
        self._handle_cell.handle = value

    @staticmethod
    def _finalize_buffer(cell: '_HandleCell') -> None:
        """
        WHAT: 가비지 컬렉터에 의해 호출되는 리소스 해제 콜백 함수입니다.
        WHY: 텐서 객체가 메모리에서 지워질 때 참조 카운트를 감소시키고, 마지막 남은 참조가 사라질 때만 GPU 버퍼를 해제합니다.
        HOW: cell.dec_ref()가 True일 때만 _gc_queue와 _gc_queued_bytes에 추가하여 듀얼 임계치로 일괄 해제합니다.
        """
        global _gc_queued_bytes
        if cell.dec_ref():
            handle = cell.handle
            if handle is not None:
                _gc_queue.add(handle)
                _gc_queued_bytes += cell.byte_length
                cell.handle = None
                if len(_gc_queue) >= 16 or _gc_queued_bytes >= _GC_BYTE_THRESHOLD:
                    flush_gc()

    def _check_disposed(self) -> None:
        # WHAT: 텐서가 이미 해제되었는지 검사하는 내부 함수입니다.
        # WHY: 해제된 메모리에 접근해 발생하는 크래시(Use-After-Free)를 방지하기 위함입니다.
        # HOW: _disposed 플래그가 True이면 예외를 발생시킵니다.
        if self._disposed:
            raise AMEVAForgeDisposedError("Cannot access a disposed Tensor.")

    async def realize(self) -> None:
        """
        WHAT: 레이지 평가(Lazy Evaluation) 그래프를 단일 FFI 호출로 GPU에 제출(Submit)하는 함수입니다.
        WHY: 연산들을 모았다가 한 번에 수행하여 커널 호출 오버헤드를 줄이고 최적화 기회를 얻기 위함입니다.
        HOW: 위상 정렬된 노드들을 순회하며 명령(instruction) 목록을 만들고 브릿지를 통해 JS로 전달합니다.
        """
        if len(_gc_queue) > 0:
            flush_gc()
        flush_gc()
        if self.device == "cpu" or self._handle is not None:
            return

        # WHAT: 현재 텐서를 계산하기 위한 위상 정렬된 노드 리스트입니다.
        # WHY: 의존성이 없는 순서대로 노드를 처리해야 에러 없이 그래프를 빌드할 수 있기 때문입니다.
        # HOW: build_lazy_topo 함수를 호출합니다.
        topo = build_lazy_topo(self)
        from .graph import GraphBuilder
        # WHAT: JS로 넘길 연산 명령어들을 모아주는 빌더 객체입니다.
        # WHY: 복잡한 연산 그래프를 브릿지가 이해할 수 있는 평탄화된 배열 포맷으로 변환하기 위함입니다.
        # HOW: GraphBuilder 클래스의 인스턴스를 생성합니다.
        builder = GraphBuilder()

        # WHAT: 파이썬 객체의 id를 빌더의 내부 노드 ID와 매핑하는 딕셔너리입니다.
        # WHY: 텐서 간의 참조 관계를 빌더 내의 정수형 ID 기반 참조로 변환하기 위함입니다.
        # HOW: 빈 딕셔너리를 생성한 뒤 반복문에서 값을 채웁니다.
        node_id_map: dict = {}
        
        # WHAT: 위상 정렬된 텐서 목록을 순회하여 연산 그래프를 빌드하는 반복문입니다.
        # WHY: 브릿지(C/WASM)로 넘길 명령 스트림을 순서대로 생성하기 위함입니다.
        # HOW: topo 리스트 내의 각 텐서(v)를 확인하여 분기 처리합니다.
        for v in topo:
            if v._handle is not None:
                # WHAT: 노드를 불러오기(load) 위한 정수형 식별자입니다.
                # WHY: 이미 할당된 텐서를 참조하여 연산을 수행하기 위함입니다.
                # HOW: builder.add_load를 호출하여 반환된 ID를 저장합니다.
                nid = builder.add_load(v.shape, v._handle)
                node_id_map[id(v)] = nid
            elif v._lazy_op == 'upload':
                nid = builder.add_upload(v.shape, v._data)
                node_id_map[id(v)] = nid
            else:
                # WHAT: 현재 연산의 입력으로 사용될 부모 노드들의 ID 리스트입니다.
                # WHY: builder가 의존성 있는 이전 명령들을 참조할 수 있게 하기 위함입니다.
                # HOW: 부모 텐서를 순회하며 매핑된 ID를 모읍니다.
                in_ids = []
                # WHAT: 부모 텐서들을 순회하는 반복문입니다.
                # WHY: 모든 입력의 식별자를 추출하기 위함입니다.
                # HOW: v._parents 튜플을 반복합니다.
                for p in v._parents:
                    if id(p) not in node_id_map:
                        raise AMEVAForgeDeviceError(
                            f"Lazy graph build failed: parent tensor (op={p._lazy_op!r}) "
                            f"is not in the computation graph. It may have been disposed."
                        )
                    in_ids.append(node_id_map[id(p)])
                nid = builder.add_op(v._lazy_op, v.shape, in_ids, v._lazy_params)
                node_id_map[id(v)] = nid

        # WHAT: JS로 보낼 연산 명령어 배열과 추가 입력 데이터입니다.
        # WHY: 런타임에서 그래프를 재구성하여 커널을 실행하기 위한 최종 데이터 형태이기 때문입니다.
        # HOW: builder.compile()을 호출해 튜플 형태로 받습니다.
        instructions, inputs = builder.compile()
        from .bridge import js_execute_graph

        # WHAT: JS 브릿지가 연산을 실행한 뒤 반환한 새로운 텐서 핸들(버퍼 ID)들입니다.
        # WHY: 생성된 텐서 결과를 파이썬 텐서 객체와 연결(binding)하기 위함입니다.
        # HOW: js_execute_graph 함수를 호출하여 딕셔너리로 반환받습니다.
        out_handles = await js_execute_graph(instructions, inputs)

        # WHAT: 생성된 핸들들을 각 텐서 객체에 주입하는 반복문입니다.
        # WHY: 레이지(지연) 상태였던 텐서들이 이제 실제 GPU 버퍼를 가리키게 하기 위함입니다.
        # HOW: topo를 다시 순회하며 out_handles 맵에서 식별자를 찾아 할당합니다.
        for v in topo:
            if v._handle is not None:
                continue
            nid = node_id_map[id(v)]
            # WHAT: JS가 반환한 개별 텐서의 실제 문자열 핸들입니다.
            # WHY: 파이썬 쪽 텐서 객체에 이 값을 심어주기 위함입니다.
            # HOW: 반환된 딕셔너리에서 nid 키로 조회합니다.
            h = out_handles.get(str(nid)) or out_handles.get(nid)
            if h is None:
                raise AMEVAForgeDeviceError(
                    f"Failed to retrieve valid tensor handle from JS for node {nid} "
                    f"(op={v._lazy_op!r}). The JS graph executor may have failed silently."
                )
            v._handle = h
            if v._lazy_op == 'upload':
                v._data = None
                
            v._parents = ()
            v._lazy_op = None

    def numpy(self) -> np.ndarray:
        """
        WHAT: CPU 텐서의 데이터를 동기적으로 반환하는 함수입니다.
        WHY: 텐서 안의 실제 값(수치)을 확인하거나 디버깅, 외부 라이브러리(numpy 기반)에 데이터를 넘기기 위함입니다.
        HOW: 장치가 'cpu'인지 확인한 뒤 저장된 내부 _data 배열을 반환합니다.
        """
        self._check_disposed()
        if self.device == "cpu":
            if self._data is None:
                raise AMEVAForgeDisposedError("CPU tensor data has been released.")
            return self._data
        else:
            raise AMEVAForgeDeviceError(
                "GPU tensor readback is asynchronous. Use: data = await tensor.numpy_async()"
            )

    async def numpy_async(self) -> np.ndarray:
        """
        WHAT: GPU 텐서 데이터를 비동기로 읽어오는 함수입니다.
        WHY: GPU에서 CPU로 메모리를 복사하는 작업은 메인 스레드를 블로킹할 수 있으므로 비동기적으로 처리하기 위함입니다.
        HOW: JS/WebGPU의 비동기 버퍼 맵핑 기능을 활용해 완료를 대기(await)한 후 데이터를 복사합니다.
        """
        self._check_disposed()
        if self.device == "cpu":
            if self._data is None:
                raise AMEVAForgeDisposedError("CPU tensor data has been released.")
            return self._data

        # 1. 레이지 그래프를 GPU에 제출 (동기 submit -> 이제 async)
        await self.realize()

        from .bridge import js_map_async, js_read_mapped_into
        # 2. GPU 큐 완료 대기 + staging 버퍼 맵핑
        await js_map_async(self._handle)

        # 3. WASM 힙에 직접 읽어들이기
        # WHAT: 데이터를 수신할 빈 numpy 배열입니다.
        # WHY: GPU로부터 복사해올 데이터를 담아둘 메모리 공간을 미리 준비하기 위함입니다.
        # HOW: np.empty를 사용하여 텐서와 동일한 크기(shape)의 float32 배열을 할당합니다.
        out = np.empty(self.shape, dtype=np.float32)
        js_read_mapped_into(self._handle, out)

        return out

    def detach(self) -> 'Tensor':
        """
        WHAT: 연산 그래프에서 분리된 새로운 텐서(View)를 반환합니다.
        WHY: 기존 데이터/GPU 버퍼 저장소(_handle_cell)를 참조 카운트 기반으로 안전하게 공유하면서 그래디언트 추적(Autograd)을 중단하기 위함입니다.
        HOW: self._handle_cell의 참조 카운트를 1 올리고 handle_cell로 전달하여 동일 버퍼를 안전하게 바인딩합니다.
        """
        self._check_disposed()
        out = Tensor(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
            handle=self._handle,
            data=self._data,
            handle_cell=self._handle_cell if self.device == "gpu" else None,
        )
        out._disposed = self._disposed
        out._parents = ()
        out._lazy_op = None
        out._grad_parents = ()
        out._ctx = None
        return out

    def to(self, device: str) -> 'Tensor':
        """
        WHAT: CPU Tensor를 GPU lazy-upload Tensor로 이동하거나 장치를 변경합니다.
        WHY: 텐서 데이터를 GPU 메모리로 업로드하기 위해 지연 업로드 노드를 생성합니다.
        HOW: 새로운 GPU 장치 Tensor 객체를 생성하여 반환합니다.
        """
        self._check_disposed()
        if device not in ("cpu", "gpu"):
            raise AMEVAForgeDeviceError(f"Unsupported device: {device}")
        if device == self.device:
            return self
        if device == "cpu":
            if self._handle is not None:
                raise AMEVAForgeDeviceError(
                    "GPU to CPU transfer is asynchronous. Use await tensor.numpy_async()."
                )
            if self._data is not None:
                return Tensor(
                    shape=self.shape,
                    dtype=self.dtype,
                    device="cpu",
                    requires_grad=self.requires_grad,
                    data=self._data.astype(np.float32, copy=True),
                )
            raise AMEVAForgeDeviceError("Cannot move unallocated GPU tensor to CPU synchronously")
        if self._data is None:
            raise AMEVAForgeDeviceError("CPU tensor has no uploadable data")

        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            device="gpu",
            requires_grad=self.requires_grad,
            data=self._data.astype(np.float32, copy=True),
            op="upload",
            parents=(),
            op_params=[],
        )

    def move_to_(self, device: str) -> 'Tensor':
        """
        WHAT: Tensor의 내부 저장소(Storage/Device)를 in-place로 대상 장치로 마이그레이션합니다.
        WHY: Module.to(device) 호출 시 Parameter 객체 참조(Identity)를 보존하여,
             Optimizer 생성 후 model.to('gpu')를 호출해도 가중치 업데이트 참조가 단절되지 않도록 하기 위함입니다.
        HOW: self.to(device)로 상태를 생성한 후, self의 내부 속성들을 in-place 덮어쓰고 moved의 소유권을 박탈합니다.
        """
        self._check_disposed()
        if device == self.device:
            return self

        moved = self.to(device)

        # 기존 장치 리소스 정리
        if self.device == "gpu" and getattr(self, "_handle_cell", None) is not None:
            try:
                self.dispose()
            except Exception:
                pass

        # moved의 내부 상태를 self로 in-place 이전
        self._data = moved._data
        self._handle_cell = getattr(moved, "_handle_cell", None)
        if device == "gpu" and self._handle_cell is not None:
            self._handle_cell.inc_ref()
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)

        self._lazy_op = getattr(moved, "_lazy_op", None)
        self._op = getattr(moved, "_op", None)
        self._parents = getattr(moved, "_parents", ())
        self._op_params = getattr(moved, "_op_params", None)
        self.shape = moved.shape
        self.dtype = moved.dtype
        self.device = moved.device
        self._disposed = False
        self._finalizer_registered = (device == "gpu")
        self._version += 1

        moved._data = None
        moved._handle_cell = None

        return self

    def dispose(self) -> None:
        """
        WHAT: 텐서와 연결된 리소스(GPU 버퍼 및 내부 데이터)를 즉시 해제하는 함수입니다.
        WHY: 더 이상 사용하지 않는 메모리를 명시적으로 반환하여 VRAM 초과(OOM) 오류를 막기 위함입니다.
        HOW: _handle_cell의 참조 카운트를 감소시키고, 마지막 참조일 때만 GC 큐에 등록하여 flush_gc()를 호출합니다.
        """
        if self._disposed:
            return
        if self.device == "gpu" and self._handle_cell is not None:
            if self._handle_cell.dec_ref():
                handle = self._handle_cell.handle
                if handle is not None:
                    global _gc_queued_bytes
                    _gc_queue.add(handle)
                    _gc_queued_bytes += self._handle_cell.byte_length
                    self._handle_cell.handle = None
                    flush_gc()

        self._data = None
        self._disposed = True
        self._lazy_op = None
        self._parents = ()
        self._grad_parents = ()
        self._ctx = None

    def backward(self, gradient: Optional['Tensor'] = None, retain_graph: bool = False) -> None:
        """
        WHAT: 역전파(Backpropagation)를 수행하여 이 텐서에 기여한 모든 부모 텐서들의 기울기(gradient)를 계산하는 함수입니다.
        WHY: 신경망을 학습시킬 때 손실 함수(Loss)로부터 파라미터 업데이트에 필요한 미분값을 구하기 위해서입니다.
        HOW: 위상 정렬의 역순으로 그래프를 탐색하며, 연산 클래스의 backward 함수에 체인 룰(Chain Rule)을 적용합니다.
        """
        self._check_disposed()
        if retain_graph:
            from .errors import AMEVAForgeUnsupportedOperationError
            raise AMEVAForgeUnsupportedOperationError(
                "retain_graph=True is outside Release 1"
            )
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require grad.")

        from .autograd import build_topological_sort
        from .ops import ones_like

        if gradient is None:
            if self.shape != () and self.shape != (1,):
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            else:
                gradient = ones_like(self)

        # WHAT: 각 텐서 노드의 ID를 키로 하여 해당 노드까지 누적된 기울기를 저장하는 맵입니다.
        # WHY: 중복된 경로를 통해 전달되는 기울기들을 하나로 모아(add) 기록하기 위함입니다.
        # HOW: 현재 노드(루트)에 대한 기울기를 초기화하여 딕셔너리에 넣습니다.
        grads: dict = {id(self): gradient}

        # WHAT: 순전파 시 생성된 그래프를 탐색하기 위한 위상 정렬된 노드 리스트입니다.
        # WHY: 계산 순서를 보장해야 부모 노드로 기울기를 올바르게 전달할 수 있기 때문입니다.
        # HOW: autograd 모듈의 함수를 호출합니다.
        topo = build_topological_sort(self)
        
        # WHAT: 정렬된 노드를 역방향으로 순회하며 역전파를 수행하는 반복문입니다.
        # WHY: 체인 룰에 따라 최종 출력(현재 텐서)에서부터 입력(부모) 방향으로 기울기를 흘려보내야 하기 때문입니다.
        # HOW: reversed() 함수를 통해 리스트의 순서를 뒤집어 반복합니다.
        for v in reversed(topo):
            if not getattr(v, 'requires_grad', False):
                continue
            if id(v) not in grads:
                continue

            # WHAT: 현재 노드(v)에 전달된 누적 기울기입니다.
            # WHY: 부모 노드들의 기울기를 구하기 위해 이 값을 곱해야(Chain Rule) 하기 때문입니다.
            # HOW: grads 맵에서 꺼내오고 메모리 해제를 위해 pop합니다.
            grad_out = grads.pop(id(v))

            if v._ctx is None or v._op_cls is None:
                if v.grad is None:
                    v.grad = grad_out
                else:
                    from .ops import add
                    v.grad = add(v.grad, grad_out)
                continue

            v._ctx.validate_saved_tensor_versions()
            grad_inputs = v._op_cls.backward(v._ctx, grad_out)
            v._ctx.saved_tensors = ()
            v._ctx.saved_versions = ()
            if not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)

            # WHAT: 계산된 부모 기울기들을 각 부모 노드에 전달하는 반복문입니다.
            # WHY: 부모 노드의 ID 맵(grads)에 값을 누적(add)하기 위함입니다.
            # HOW: _grad_parents 튜플과 grad_inputs를 zip으로 묶어 순회합니다.
            for parent, g in zip(v._grad_parents, grad_inputs):
                if not getattr(parent, 'requires_grad', False) or g is None:
                    continue
                # WHAT: 부모 노드의 메모리 주소(ID)입니다.
                # WHY: grads 딕셔너리에 접근하는 식별자로 사용하기 위함입니다.
                # HOW: id() 함수를 사용합니다.
                pid = id(parent)
                if pid in grads:
                    from .ops import add
                    grads[pid] = add(grads[pid], g)
                else:
                    grads[pid] = g
        
        # PyTorch Autograd Engine 표준: 역전파 완료 즉시 DAG 부모 참조와 Context를 소각하여 순환 참조와 VRAM 누수를 방지
        for node in topo:
            node._grad_parents = ()
            node._ctx = None
        grads.clear()



    def relu(self) -> 'Tensor':
        """
        WHAT: 현재 텐서에 ReLU(Rectified Linear Unit) 활성화 함수를 적용합니다.
        WHY: 신경망에 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 하기 위함입니다.
        HOW: ops 모듈의 relu 연산을 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import relu
        return relu(self)

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        WHAT: 현재 텐서와 다른 텐서 간의 행렬 곱셈을 수행합니다.
        WHY: 선형 계층(Linear Layer) 등에서 가중치와의 곱 연산을 처리하기 위함입니다.
        HOW: ops 모듈의 matmul 함수를 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import matmul
        return matmul(self, other)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        WHAT: 파이썬의 `@` 연산자를 오버로딩하여 행렬 곱셈을 수행합니다.
        WHY: 사용자가 `a @ b`처럼 직관적으로 수학 기호를 사용할 수 있게 하기 위함입니다.
        HOW: 내부적으로 self.matmul()을 호출합니다.
        """
        return self.matmul(other)

    def __add__(self, other):
        """
        WHAT: 두 텐서 간의 덧셈 연산(`+`)을 수행합니다.
        WHY: 텐서들 간의 요소별 덧셈을 직관적으로 지원하기 위함입니다.
        HOW: 스칼라 값일 경우 () 크기의 스칼라 텐서로 변환하여 0-stride 브로드캐스팅으로 ops.add를 호출합니다.
        """
        self._check_disposed()
        from .ops import add, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return add(self, other)

    def __radd__(self, other):
        """
        WHAT: 우측 피연산자 기준의 덧셈 연산(`스칼라 + 텐서`)을 수행합니다.
        WHY: 파이썬 내장 스칼라 타입이 왼쪽에 올 때도 덧셈이 동작하도록 지원하기 위함입니다.
        HOW: 덧셈은 교환 법칙이 성립하므로 self.__add__(other)를 그대로 반환합니다.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        WHAT: 두 텐서 간의 뺄셈 연산(`-`)을 수행합니다.
        WHY: 텐서 요소들 간의 차이를 계산하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 변환한 뒤 0-stride 브로드캐스팅으로 ops.sub를 호출합니다.
        """
        self._check_disposed()
        from .ops import sub, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return sub(self, other)

    def __rsub__(self, other):
        """
        WHAT: 우측 피연산자 기준의 뺄셈 연산(`스칼라 - 텐서`)을 수행합니다.
        WHY: 스칼라가 왼쪽에 올 경우 순서에 맞게 뺄셈을 적용하기 위함입니다.
        HOW: other를 () 스칼라 텐서로 변환한 뒤 other에서 self를 빼는 ops.sub를 호출합니다.
        """
        self._check_disposed()
        from .ops import sub, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return sub(other, self)

    def __mul__(self, other):
        """
        WHAT: 두 텐서 간의 요소별 곱셈 연산(`*`)을 수행합니다.
        WHY: 아다마르 곱(Hadamard Product)이나 스칼라 배율을 적용하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 변환 후 0-stride 브로드캐스팅으로 ops.mul을 호출합니다.
        """
        self._check_disposed()
        from .ops import mul, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return mul(self, other)

    def __rmul__(self, other):
        """
        WHAT: 우측 피연산자 기준의 곱셈 연산(`스칼라 * 텐서`)을 수행합니다.
        WHY: 곱셈의 교환 법칙을 지원하여 스칼라가 왼쪽에 와도 처리되게 하기 위함입니다.
        HOW: self.__mul__(other)를 호출하여 결과를 반환합니다.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        WHAT: 두 텐서 간의 나눗셈 연산(`/`)을 수행합니다.
        WHY: 텐서 요소별 나눗셈을 수식으로 간편하게 표현하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 변환 후 0-stride 브로드캐스팅으로 ops.div를 호출합니다.
        """
        self._check_disposed()
        from .ops import div, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return div(self, other)

    def __rtruediv__(self, other):
        """
        WHAT: 우측 피연산자 기준의 나눗셈 연산(`스칼라 / 텐서`)을 수행합니다.
        WHY: 스칼라를 텐서의 각 요소로 나누는 연산을 지원하기 위함입니다.
        HOW: 스칼라를 () 스칼라 텐서로 바꾼 후 other를 self로 나누는 ops.div를 호출합니다.
        """
        self._check_disposed()
        from .ops import div, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return div(other, self)

    def __neg__(self):
        """
        WHAT: 텐서의 부호를 반전시키는 단항 연산(`-텐서`)을 수행합니다.
        WHY: 수식 내에서 값의 부호를 쉽게 바꾸기 위함입니다.
        HOW: ops.neg 함수를 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import neg
        return neg(self)

    def __pow__(self, exponent):
        """
        WHAT: 거듭제곱 연산(`x ** exponent`)을 수행합니다.
        WHY: 지수 계산 및 L2 정규화, 제곱근 등을 편리하게 계산하기 위함입니다.
        HOW: ops.pow_op 함수를 호출하여 Zero-Safe 미분이 지원되는 텐서를 반환합니다.
        """
        self._check_disposed()
        from .ops import pow_op
        return pow_op(self, exponent)

    def clone(self) -> 'Tensor':
        """
        WHAT: 텐서의 복제본을 생성하여 반환합니다.
        WHY: 원본 텐서와 분리된 메모리를 가지면서도 역전파 그래프는 유지하기 위함입니다.
        HOW: ops.clone 함수를 호출하여 복제 텐서를 생성합니다.
        """
        self._check_disposed()
        from .ops import clone
        return clone(self)

    # =========================================================================
    # PyTorch 호환 In-place 연산자 및 Autograd 버전 관리 (_version increment)
    # =========================================================================
    def __iadd__(self, other):
        """In-place addition (self += other) with version increment."""
        return self.add_(other)

    def add_(self, other):
        """
        WHAT: 현재 텐서에 other를 in-place 덧셈하고 버전을 증가시킵니다.
        WHY: 메모리 재할당 없이 텐서를 수정하며, 역전파 시 saved_tensors 변조를 감지하기 위함입니다.
        """
        self._check_disposed()
        from .ops import add, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        if self.device == 'cpu':
            res = add(self, other)
            if res._data is not None:
                self._data = res._data
        else:
            old_self = Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                handle=self._handle,
                data=self._data,
                op=self._lazy_op,
                parents=self._parents,
                op_params=self._lazy_params,
                handle_cell=self._handle_cell,
            )
            res = add(old_self, other)
            self._lazy_op = res._lazy_op
            self._parents = res._parents
            self._lazy_params = res._lazy_params
            self._handle_cell = _HandleCell(None, old_self._handle_cell.byte_length)
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
        self._version += 1
        return self

    def __isub__(self, other):
        """In-place subtraction (self -= other) with version increment."""
        return self.sub_(other)

    def sub_(self, other):
        """
        WHAT: 현재 텐서에 other를 in-place 뺄셈하고 버전을 증가시킵니다.
        """
        self._check_disposed()
        from .ops import sub, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        if self.device == 'cpu':
            res = sub(self, other)
            if res._data is not None:
                self._data = res._data
        else:
            old_self = Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                handle=self._handle,
                data=self._data,
                op=self._lazy_op,
                parents=self._parents,
                op_params=self._lazy_params,
                handle_cell=self._handle_cell,
            )
            res = sub(old_self, other)
            self._lazy_op = res._lazy_op
            self._parents = res._parents
            self._lazy_params = res._lazy_params
            self._handle_cell = _HandleCell(None, old_self._handle_cell.byte_length)
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
        self._version += 1
        return self

    def __imul__(self, other):
        """In-place multiplication (self *= other) with version increment."""
        return self.mul_(other)

    def mul_(self, other):
        """
        WHAT: 현재 텐서에 other를 in-place 곱셈하고 버전을 증가시킵니다.
        """
        self._check_disposed()
        from .ops import mul, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        if self.device == 'cpu':
            res = mul(self, other)
            if res._data is not None:
                self._data = res._data
        else:
            old_self = Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                handle=self._handle,
                data=self._data,
                op=self._lazy_op,
                parents=self._parents,
                op_params=self._lazy_params,
                handle_cell=self._handle_cell,
            )
            res = mul(old_self, other)
            self._lazy_op = res._lazy_op
            self._parents = res._parents
            self._lazy_params = res._lazy_params
            self._handle_cell = _HandleCell(None, old_self._handle_cell.byte_length)
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
        self._version += 1
        return self

    def __itruediv__(self, other):
        """In-place division (self /= other) with version increment."""
        return self.div_(other)

    def div_(self, other):
        """
        WHAT: 현재 텐서에 other를 in-place 나눗셈하고 버전을 증가시킵니다.
        """
        self._check_disposed()
        from .ops import div, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        if self.device == 'cpu':
            res = div(self, other)
            if res._data is not None:
                self._data = res._data
        else:
            old_self = Tensor(
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
                handle=self._handle,
                data=self._data,
                op=self._lazy_op,
                parents=self._parents,
                op_params=self._lazy_params,
                handle_cell=self._handle_cell,
            )
            res = div(old_self, other)
            self._lazy_op = res._lazy_op
            self._parents = res._parents
            self._lazy_params = res._lazy_params
            self._handle_cell = _HandleCell(None, old_self._handle_cell.byte_length)
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
        self._version += 1
        return self

    def fill_(self, value: float):
        """
        WHAT: 현재 텐서를 지정된 상수 값으로 in-place 채우고 버전을 증가시킵니다.
        WHY: 기존 버퍼를 안전하게 해제/치환하고 새 상수 연산 노드를 연결하기 위함입니다.
        """
        self._check_disposed()
        if self.device == 'cpu':
            if self._data is not None:
                self._data.fill(float(value))
        else:
            if self._handle_cell is not None and self._handle_cell.handle is not None:
                try:
                    self.dispose()
                except Exception:
                    pass
            from .ops import full
            res = full(self.shape, float(value), dtype=self.dtype, device='gpu')
            self._lazy_op = res._lazy_op
            self._parents = res._parents
            self._lazy_params = res._lazy_params
            self._handle_cell = _HandleCell(None, res._handle_cell.byte_length)
            self._disposed = False
            import weakref
            weakref.finalize(self, Tensor._finalize_buffer, self._handle_cell)
        self._version += 1
        return self

    def zero_(self):
        """In-place zero out the tensor."""
        return self.fill_(0.0)

    def clamp(self, min_val=None, max_val=None):
        """Clamps all elements in input into the range [min_val, max_val]."""
        self._check_disposed()
        from .ops import clamp
        return clamp(self, min_val, max_val)

    def maximum(self, other):
        """Element-wise maximum of self and other."""
        self._check_disposed()
        from .ops import maximum, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return maximum(self, other)

    def minimum(self, other):
        """Element-wise minimum of self and other."""
        self._check_disposed()
        from .ops import minimum, tensor
        if isinstance(other, (int, float)):
            other = tensor(float(other), device=self.device, dtype=self.dtype)
        return minimum(self, other)

    def triu(self, diagonal: int = 0):
        """Returns the upper triangular part of a matrix or batch of matrices."""
        self._check_disposed()
        from .ops import triu
        return triu(self, diagonal)

    def tril(self, diagonal: int = 0):
        """Returns the lower triangular part of a matrix or batch of matrices."""
        self._check_disposed()
        from .ops import tril
        return tril(self, diagonal)

    def gelu(self, approximate: str = "none"):
        """Applies GELU to this tensor."""
        self._check_disposed()
        from .ops import gelu
        return gelu(self, approximate=approximate)

    def silu(self):
        """Applies SiLU (Swish) to this tensor."""
        self._check_disposed()
        from .ops import silu
        return silu(self)

    def leaky_relu(self, negative_slope: float = 0.01):
        """Applies LeakyReLU to this tensor."""
        self._check_disposed()
        from .ops import leaky_relu
        return leaky_relu(self, negative_slope=negative_slope)

    def elu(self, alpha: float = 1.0):
        """Applies ELU to this tensor."""
        self._check_disposed()
        from .ops import elu
        return elu(self, alpha=alpha)

    def sum(self):
        """
        WHAT: 텐서 내 모든 요소들의 합을 계산합니다.
        WHY: 손실(Loss) 합산이나 특정 차원의 데이터를 집계하기 위함입니다.
        HOW: ops.sum_op 함수를 호출하여 스칼라(또는 축소된 텐서)를 반환합니다.
        """
        self._check_disposed()
        from .ops import sum_op
        return sum_op(self)

    def mean(self):
        """
        WHAT: 텐서 내 모든 요소들의 평균을 계산합니다.
        WHY: MSE 손실 계산 등에서 전체 데이터의 평균적인 크기를 구하기 위함입니다.
        HOW: ops.mean_op 함수를 호출하여 결과를 반환합니다.
        """
        self._check_disposed()
        from .ops import mean_op
        return mean_op(self)

    def reshape(self, *shape):
        """
        WHAT: 텐서의 차원 형태(Shape)를 변경합니다.
        WHY: 데이터의 논리적 구조를 재배열하여 다른 연산(행렬곱 등)과 호환되게 하기 위함입니다.
        HOW: 새로운 shape 튜플을 구성하고 ops.reshape를 호출합니다.
        """
        self._check_disposed()
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def view(self, *shape):
        """
        WHAT: reshape와 동일하게 텐서의 형태를 변경합니다.
        WHY: 기존 파이토치(PyTorch) 코드와의 호환성을 유지하여 사용 편의성을 높이기 위함입니다.
        HOW: 내부적으로 self.reshape를 호출합니다.
        """
        return self.reshape(*shape)

    def numel(self):
        """
        WHAT: 텐서가 포함하는 전체 원소(Element)의 개수를 반환합니다.
        WHY: 데이터의 총 크기를 파악하거나 평균 계산 시 나누는 값으로 사용하기 위함입니다.
        HOW: 텐서의 shape 튜플을 순회하며(loop) 각 차원의 크기를 곱합니다.
        """
        # WHAT: 원소 개수를 누적해서 곱할 변수입니다.
        # WHY: 1부터 시작하여 각 차원을 곱해가기 위함입니다.
        # HOW: 정수 1로 초기화합니다.
        n = 1
        # WHAT: shape의 각 차원 크기를 순회하는 반복문입니다.
        # WHY: 모든 차원의 크기를 곱하여 총 볼륨을 구하기 위함입니다.
        # HOW: self.shape를 순회하여 n에 곱합니다.
        for d in self.shape:
            n *= d
        return n

    @property
    def data(self):
        """
        WHAT: 현재 텐서 객체 자체를 반환합니다 (데이터 속성 접근용).
        WHY: 파이토치 스타일의 tensor.data 접근 패턴을 모방하여 기존 코드 호환성을 제공하기 위함입니다.
        HOW: self를 그대로 반환합니다.
        """
        return self

    @data.setter
    def data(self, new_tensor):
        """
        WHAT: 텐서의 내부 데이터와 상태를 다른 텐서의 것으로 교체(In-place replacement)합니다.
        WHY: 옵티마이저(Optimizer)가 파라미터를 업데이트할 때 새로운 텐서를 할당하지 않고 제자리에서 값을 바꾸기 위함입니다.
        HOW: 새 텐서의 데이터, 핸들, 형태를 복사하고, 기존의 자동 미분 그래프 연결(부모, 컨텍스트 등)을 모두 끊습니다.
        """
        # In-place replacement for optimizer updates
        self._data = new_tensor._data if hasattr(new_tensor, '_data') else None
        self.shape = new_tensor.shape
        self._handle = getattr(new_tensor, '_handle', None)
        self._parents = ()
        self._grad_parents = ()  # VUL-005 Fix: autograd 그래프 참조 해제
        self._ctx = None         # VUL-005 Fix: backward context 해제
        self._op_cls = None      # VUL-005 Fix: op class 해제
        self._grad_fn = None
        self.grad = None
        
    def exp(self):
        """
        WHAT: 텐서의 각 요소에 지수 함수(e^x)를 적용합니다.
        WHY: 소프트맥스(Softmax) 등의 활성화 함수나 로그 확정값을 원래 스케일로 복원하기 위함입니다.
        HOW: ops.exp_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import exp_op
        return exp_op(self)
        
    def log(self):
        """
        WHAT: 텐서의 각 요소에 자연 로그(ln x)를 적용합니다.
        WHY: 크로스 엔트로피 손실(Cross Entropy Loss) 등 정보량 기반 수식 처리를 위함입니다.
        HOW: ops.log_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import log_op
        return log_op(self)
        
    def sigmoid(self):
        """
        WHAT: 시그모이드(Sigmoid) 활성화 함수를 적용합니다.
        WHY: 출력값을 0과 1 사이의 확률값으로 변환하기 위함입니다.
        HOW: ops.sigmoid를 호출합니다.
        """
        self._check_disposed()
        from .ops import sigmoid
        return sigmoid(self)
        
    def tanh(self):
        """
        WHAT: 하이퍼볼릭 탄젠트(Tanh) 활성화 함수를 적용합니다.
        WHY: 출력값을 -1과 1 사이로 정규화하여 기울기 소실을 완화하기 위함입니다.
        HOW: ops.tanh_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import tanh_op
        return tanh_op(self)

    def unsqueeze(self, dim: int):
        """
        WHAT: 지정한 차원(dim) 위치에 크기가 1인 새로운 차원을 삽입합니다.
        WHY: 브로드캐스팅(Broadcasting)을 위해 차원 수를 늘리거나 배치 차원을 추가하기 위함입니다.
        HOW: ops.unsqueeze를 호출합니다.
        """
        self._check_disposed()
        from .ops import unsqueeze
        return unsqueeze(self, dim)
        
    def squeeze(self, dim: Optional[int] = None):
        """
        WHAT: 크기가 1인 차원을 제거합니다. dim 지정 시 해당 차원만 제거합니다.
        WHY: 불필요한 차원을 축소시켜 데이터 구조를 단순화하거나 차원 수를 맞추기 위함입니다.
        HOW: ops.squeeze를 호출합니다.
        """
        self._check_disposed()
        from .ops import squeeze
        return squeeze(self, dim)
        
    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        """
        WHAT: 다차원 텐서를 1차원 배열(또는 지정된 차원 구간 축소)로 평탄화합니다.
        WHY: 합성곱(CNN) 층의 출력을 선형(Linear) 층에 전달하기 위해 1차원 벡터로 펴주어야 하기 때문입니다.
        HOW: ops.flatten을 호출합니다.
        """
        self._check_disposed()
        from .ops import flatten
        return flatten(self, start_dim, end_dim)
        
    def permute(self, *dims):
        """
        WHAT: 텐서의 차원 순서를 지정된 배열(dims)대로 재배치합니다.
        WHY: 이미지 데이터의 채널 순서 변경(NHWC <-> NCHW)이나 어텐션 헤드 축 전치 등을 수행하기 위함입니다.
        HOW: ops.permute를 호출합니다. 가변 인자와 튜플/리스트 전달을 모두 지원합니다.
        """
        self._check_disposed()
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        from .ops import permute
        return permute(self, dims)

    def max(self, axis=None):
        """
        WHAT: 텐서 내 요소들의 최댓값을 구합니다. 특정 축(axis)이 주어지면 해당 축을 따라 최댓값을 구합니다.
        WHY: 예측 클래스를 선택(Argmax 역할)하거나 풀링 연산을 수행하기 위함입니다.
        HOW: axis 여부에 따라 ops.max_op 또는 ops.max_axis를 호출합니다.
        """
        self._check_disposed()
        if axis is None:
            from .ops import max_op
            return max_op(self)
        else:
            from .ops import max_axis
            return max_axis(self, axis)

    def var(self, axis=None, unbiased=True):
        """
        WHAT: 텐서의 분산(Variance)을 계산합니다.
        WHY: 데이터의 분포 범위를 파악하거나 정규화(Normalization) 과정에서 쓰이기 때문입니다.
        HOW: ops.var를 호출합니다.
        """
        self._check_disposed()
        from .ops import var
        return var(self, axis, unbiased)
        
    def std(self, axis=None, unbiased=True):
        """
        WHAT: 텐서의 표준 편차(Standard Deviation)를 계산합니다.
        WHY: 데이터의 퍼짐 정도를 원래 단위로 확인하거나 표준화 기법에서 분모로 사용하기 위함입니다.
        HOW: ops.std를 호출합니다.
        """
        self._check_disposed()
        from .ops import std
        return std(self, axis, unbiased)

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
        """Returns the k largest (or smallest) elements along dim."""
        self._check_disposed()
        from .ops import topk
        return topk(self, k, dim=dim, largest=largest, sorted=sorted)

    def sort(self, dim: int = -1, descending: bool = False, stable: bool = False):
        """Sorts the elements of the tensor along dim."""
        self._check_disposed()
        from .ops import sort
        return sort(self, dim=dim, descending=descending, stable=stable)

    def argsort(self, dim: int = -1, descending: bool = False, stable: bool = False):
        """Returns indices that sort the tensor along dim."""
        self._check_disposed()
        from .ops import argsort
        return argsort(self, dim=dim, descending=descending, stable=stable)

    def __getitem__(self, key):
        """
        WHAT: 인덱싱 또는 슬라이싱 문법(예: tensor[0:2])을 사용하여 텐서의 부분 배열을 추출합니다.
        WHY: 특정 데이터 샘플을 선택하거나 관심 영역(ROI)만 잘라내어 처리하기 위함입니다.
        HOW: 파이썬 특수 메서드를 오버로딩하여 ops.slice_op를 호출합니다.
        """
        self._check_disposed()
        from .ops import slice_op
        return slice_op(self, key)

    def __setitem__(self, key, value):
        """
        WHAT: 인덱싱 문법(예: tensor[0] = 5.0)을 사용하여 텐서의 특정 위치 값을 변경합니다.
        WHY: 배열 슬라이스 갱신 및 마스킹 처리를 수행하기 위함입니다.
        HOW: CPU 텐서인 경우 내부 버퍼를 수정하고 버전을 증가시키며, GPU 텐서인 경우 scatter 또는 CPU 전이를 안내합니다.
        """
        self._check_disposed()
        if self.device == 'cpu' and self._data is not None:
            if isinstance(value, Tensor):
                self._data[key] = value.numpy()
            else:
                self._data[key] = value
            self._version += 1
        else:
            raise NotImplementedError("Direct in-place item assignment on GPU tensors is not supported. Transfer to CPU or use functional scatter.")

    def __repr__(self) -> str:
        """
        WHAT: 텐서 객체를 문자열로 표현(출력)합니다.
        WHY: 디버깅이나 로그 확인 시 텐서의 크기, 타입, 디바이스 등 메타데이터를 쉽게 확인하기 위함입니다.
        HOW: 포맷팅된 문자열을 생성하여 반환합니다. 해제된 텐서는 별도로 표시합니다.
        """
        if self._disposed:
            return '<AMEVA Tensor (disposed)>'
        return (
            f'<AMEVA Tensor shape={self.shape}, dtype={self.dtype}, '
            f'device={self.device}, requires_grad={self.requires_grad}, '
            f'handle={self._handle}>'
        )
