from typing import Any, Tuple, Optional, List
import contextlib

_grad_mode = True


@contextlib.contextmanager
def no_grad():
    global _grad_mode
    prev = _grad_mode
    _grad_mode = False
    try:
        yield
    finally:
        _grad_mode = prev


class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *args):
        self.saved_tensors = args


class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        """Forward pass를 실행하고 autograd 그래프를 구성한다."""
        requires_grad = _grad_mode and any(
            (hasattr(a, 'requires_grad') and a.requires_grad) for a in args
        )

        ctx = Context()
        result = cls.forward(ctx, *args, **kwargs)

        if requires_grad:
            result.requires_grad = True
            result._ctx = ctx
            result._op_cls = cls
            # NC-05 Fix: autograd 부모를 _grad_parents에 저장 (레이지 그래프의 _parents를 덮어쓰지 않음)
            # NH-08 Fix: isinstance 대신 hasattr로 Tensor 여부 확인 (서브클래스도 처리)
            result._grad_parents = tuple(
                a for a in args
                if hasattr(a, 'requires_grad') and hasattr(a, 'shape')
            )

        return result

    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, ...]:
        raise NotImplementedError


# VUL-009: 그래프 노드 수 제한 설정
_max_graph_nodes_warning: int = 10000
_max_graph_nodes_hard_limit: Optional[int] = None  # None = 무제한


def set_max_graph_nodes(warning: int = 10000, hard_limit: Optional[int] = None) -> None:
    """
    Autograd 그래프 노드 수 제한을 설정한다.
    warning: 이 값을 초과하면 RuntimeWarning 발생 (기본 10000)
    hard_limit: 이 값을 초과하면 RuntimeError 발생 (None = 무제한)
    """
    global _max_graph_nodes_warning, _max_graph_nodes_hard_limit
    _max_graph_nodes_warning = warning
    _max_graph_nodes_hard_limit = hard_limit


def build_topological_sort(root) -> List[Any]:
    """
    NH-08 Fix: id(v) 기반으로 visited 추적 (PyTorch, JAX 방식).
    Iterative DFS — 재귀 제거 (RecursionError 방지).
    backward를 위한 _grad_parents를 사용하여 그래프 탐색.
    VUL-009: 그래프 노드 수 warning/hard limit.
    """
    topo = []
    visited = set()
    stack = [(root, 0)]
    visited.add(id(root))
    _warned = False

    while stack:
        node, idx = stack[-1]
        grad_parents = getattr(node, '_grad_parents', ())
        if idx < len(grad_parents):
            stack[-1] = (node, idx + 1)
            p = grad_parents[idx]
            pid = id(p)
            if pid not in visited:
                visited.add(pid)
                stack.append((p, 0))

                # VUL-009: 노드 수 제한 체크
                node_count = len(visited)
                if _max_graph_nodes_hard_limit is not None and node_count > _max_graph_nodes_hard_limit:
                    raise RuntimeError(
                        f"Autograd graph exceeds hard limit of {_max_graph_nodes_hard_limit} nodes. "
                        f"Use set_max_graph_nodes(hard_limit=N) to adjust."
                    )
                if not _warned and node_count > _max_graph_nodes_warning:
                    import warnings
                    warnings.warn(
                        f"[AMEVA] Autograd graph has {node_count} nodes (warning threshold: "
                        f"{_max_graph_nodes_warning}). This may consume excessive memory. "
                        f"Consider checkpointing or reducing computation depth.",
                        RuntimeWarning, stacklevel=2
                    )
                    _warned = True
        else:
            stack.pop()
            topo.append(node)

    return topo

