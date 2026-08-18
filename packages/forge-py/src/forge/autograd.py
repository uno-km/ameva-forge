"""
=============================================================================
[파일 이력 메타데이터]
- Created: 2026-08-12 12:14:52 +0900
- Modified:
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
=============================================================================
자동 미분(Autograd) 엔진의 핵심 컴포넌트를 정의하는 모듈입니다.
"""
from typing import Any, Tuple, Optional, List
import contextlib

# _grad_mode는 현재 시스템이 그래디언트를 추적하고 있는지 여부를 나타내는 전역 상태 변수입니다.
# 기본값은 True이며, 그래디언트 계산이 필요한 연산들이 이 값을 참조하여 연산 그래프를 구성할지 결정합니다.
_grad_mode = True


@contextlib.contextmanager
def no_grad():
    """
    [WHAT] 
    그래디언트 추적을 임시로 비활성화하는 컨텍스트 매니저입니다.
    
    [WHY] 
    추론(Inference) 단계나 그래디언트 업데이트 시 불필요한 메모리 사용과 연산 오버헤드를 방지하기 위해 존재합니다.
    
    [HOW] 
    진입 시 전역 상태 변수인 _grad_mode의 기존 값을 저장하고 False로 변경한 뒤, 블록이 끝나면 원래 값으로 복원하는 방식으로 동작합니다.
    """
    # 전역 상태 변수를 수정하기 위해 global 키워드를 사용합니다.
    global _grad_mode
    # 현재 _grad_mode 값을 백업 변수인 prev에 저장합니다.
    prev = _grad_mode
    # 그래디언트 추적을 비활성화하기 위해 _grad_mode를 False로 설정합니다.
    _grad_mode = False
    try:
        # yield를 통해 제어권을 컨텍스트 블록 내부로 넘깁니다.
        yield
    finally:
        # 컨텍스트 블록의 실행이 끝나거나 예외가 발생하더라도 _grad_mode를 원래 상태로 반드시 복원합니다.
        _grad_mode = prev


class Context:
    """
    [WHAT] 
    순전파(Forward pass) 중에 계산된 중간 결과물들을 역전파(Backward pass)를 위해 저장하는 객체입니다.
    
    [WHY] 
    자동 미분을 수행하려면 순전파 때 사용된 입력 텐서나 중간값이 역전파의 그래디언트 계산 식에 필요한 경우가 많으므로 이를 안전하게 보관해야 합니다.
    
    [HOW] 
    인스턴스 변수인 saved_tensors 튜플에 필요한 데이터를 참조로 유지하며, save_for_backward 메서드를 통해 데이터를 저장합니다.
    """
    def __init__(self):
        """
        [WHAT] 
        Context 클래스의 생성자입니다.
        
        [WHY] 
        인스턴스가 생성될 때 텐서를 저장할 빈 공간을 초기화하기 위해 필요합니다.
        
        [HOW] 
        self.saved_tensors를 빈 튜플로 초기화하여 이후 텐서들이 불변 시퀀스 형태로 저장될 수 있게 준비합니다.
        """
        # 역전파 시 사용할 텐서들을 보관할 빈 튜플을 초기화합니다.
        self.saved_tensors = ()
        self.saved_versions = ()

    def save_for_backward(self, *args):
        """
        [WHAT] 주어진 인자들과 그 시점의 버전을 역전파 계산을 위해 저장하는 메서드입니다.
        """
        self.saved_tensors = args
        self.saved_versions = tuple(
            getattr(tensor, "_version", 0)
            for tensor in args
        )

    def validate_saved_tensor_versions(self):
        """
        [WHAT] 순전파 시 저장된 텐서들이 이후 in-place 수정되었는지 검증합니다.
        """
        for tensor, saved_version in zip(self.saved_tensors, self.saved_versions):
            current_version = getattr(tensor, "_version", 0)
            if current_version != saved_version:
                raise RuntimeError(
                    "Saved tensor was modified in-place after forward execution. "
                    "Re-run forward before backward."
                )


class Function:
    """
    [WHAT] 
    모든 미분 가능한 연산(Operation)의 베이스 클래스입니다.
    
    [WHY] 
    자동 미분을 지원하는 사용자 정의 연산을 일관된 인터페이스(forward, backward, apply)로 작성할 수 있도록 표준 구조를 제공하기 위해 존재합니다.
    
    [HOW] 
    순전파는 forward를, 역전파는 backward를 오버라이드하여 구현하며, 사용자는 apply 클래스 메서드를 통해 연산을 실행합니다.
    """
    @classmethod
    def apply(cls, *args, **kwargs):
        """
        [WHAT] 
        Forward pass를 실행하고 autograd 그래프를 구성합니다.
        
        [WHY] 
        단순한 순전파 실행뿐만 아니라, 결과 텐서가 이전 텐서들과 어떻게 연결되는지(의존성)를 기록하여 연산 그래프를 만들기 위해 필요합니다.
        
        [HOW] 
        _grad_mode 상태와 인자들의 requires_grad 여부를 확인한 뒤, forward를 실행하고 결과 텐서에 Context와 부모 노드 정보를 기록하여 그래프를 연결합니다.
        """
        # _grad_mode가 켜져 있고, 인자 중 하나라도 requires_grad 속성이 True이면 현재 연산도 그래디언트를 추적해야 함을 결정합니다.
        requires_grad = _grad_mode and any(
            (hasattr(a, 'requires_grad') and a.requires_grad) for a in args
        )

        # 현재 연산의 상태를 저장할 빈 Context 객체를 생성합니다.
        ctx = Context()
        # 생성한 ctx 객체와 함께 실제 순전파 로직인 forward 메서드를 호출하여 결과를 얻어옵니다.
        result = cls.forward(ctx, *args, **kwargs)

        if requires_grad:
            # 결과 텐서가 미분 가능해야 하므로 requires_grad를 True로 설정합니다.
            result.requires_grad = True
            # 역전파 시 필요한 데이터가 담긴 ctx를 결과 텐서에 연결합니다.
            result._ctx = ctx
            # 현재 텐서를 만들어낸 연산 클래스가 무엇인지 기록합니다.
            result._op_cls = cls
            # 역전파 과정에서 부모 노드로 거슬러 올라가기 위해, 미분 가능한 입력 인자들만 추려 _grad_parents 튜플로 저장합니다.
            # NC-05 Fix: autograd 부모를 _grad_parents에 저장 (레이지 그래프의 _parents를 덮어쓰지 않음)
            # NH-08 Fix: isinstance 대신 hasattr로 Tensor 여부 확인 (서브클래스도 처리)
            result._grad_parents = tuple(
                a for a in args
                if hasattr(a, 'requires_grad') and hasattr(a, 'shape')
            )

        return result

    @staticmethod
    def forward(ctx: Context, *args, **kwargs) -> Any:
        """
        [WHAT] 
        연산의 순전파 로직을 정의하는 정적 메서드입니다.
        
        [WHY] 
        각 연산(더하기, 곱하기 등)마다 실제 수행해야 할 계산 방식을 구체적으로 명시해야 하기 때문입니다.
        
        [HOW] 
        서브클래스에서 이 메서드를 오버라이드하여 입력 텐서들을 이용해 결과를 계산하고 반환하도록 구현하며, 필요 시 ctx에 중간값을 저장합니다.
        """
        # 베이스 클래스에서는 순전파가 정의되어 있지 않으므로 NotImplementedError를 발생시킵니다.
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, ...]:
        """
        [WHAT] 
        연산의 역전파 로직을 정의하는 정적 메서드입니다.
        
        [WHY] 
        출력에 대한 그래디언트(grad_output)가 주어졌을 때, 체인 룰(Chain Rule)을 이용해 입력에 대한 그래디언트를 계산하기 위해 필요합니다.
        
        [HOW] 
        서브클래스에서 이 메서드를 오버라이드하여, ctx에 저장된 데이터와 grad_output을 조합해 각 입력에 대응하는 그래디언트 튜플을 반환하도록 구현합니다.
        """
        # 베이스 클래스에서는 역전파가 정의되어 있지 않으므로 NotImplementedError를 발생시킵니다.
        raise NotImplementedError


# VUL-009: 그래프 노드 수 제한 설정
# 메모리 누수나 무한 루프를 방지하기 위해 생성 가능한 최대 그래프 노드 수를 경고하기 위한 설정값(기본 10000)입니다.
_max_graph_nodes_warning: int = 10000
# 그래프 노드 수가 이 값을 넘으면 강제로 에러를 발생시켜 프로세스 중단을 유도하는 제한값(None이면 무제한)입니다.
_max_graph_nodes_hard_limit: Optional[int] = None  # None = 무제한


def set_max_graph_nodes(warning: int = 10000, hard_limit: Optional[int] = None) -> None:
    """
    [WHAT] 
    Autograd 그래프 노드 수 제한을 설정하는 함수입니다.
    
    [WHY] 
    사용자가 모델 크기나 메모리 한계에 맞춰 그래프 허용치를 동적으로 조절하여, OOM(Out of Memory)과 같은 치명적 오류를 사전에 예방할 수 있도록 하기 위함입니다.
    
    [HOW] 
    전역 변수 _max_graph_nodes_warning과 _max_graph_nodes_hard_limit의 값을 인자로 받은 값으로 업데이트합니다.
    """
    # 전역 변수들을 수정할 수 있도록 global로 선언합니다.
    global _max_graph_nodes_warning, _max_graph_nodes_hard_limit
    # 경고 발생 임계값을 새로운 값으로 설정합니다.
    _max_graph_nodes_warning = warning
    # 강제 종료 임계값을 새로운 값으로 설정합니다.
    _max_graph_nodes_hard_limit = hard_limit


def build_topological_sort(root) -> List[Any]:
    """
    [WHAT] 
    주어진 루트 텐서부터 시작하여 역방향 연산 그래프의 위상 정렬(Topological sort) 리스트를 구축합니다.
    
    [WHY] 
    역전파 과정은 출력 노드에서 입력 노드 방향으로 순서대로 그래디언트를 전파해야 하므로, 노드들을 올바른 순서(자식에서 부모로)로 나열하기 위해 필요합니다.
    
    [HOW] 
    반복적 깊이 우선 탐색(Iterative DFS)을 사용하여 부모 노드(_grad_parents)를 재귀 없이 순회하며, 방문이 완료된 노드를 리스트에 추가하고 반환합니다.
    
    NH-08 Fix: id(v) 기반으로 visited 추적 (PyTorch, JAX 방식).
    Iterative DFS — 재귀 제거 (RecursionError 방지).
    backward를 위한 _grad_parents를 사용하여 그래프 탐색.
    VUL-009: 그래프 노드 수 warning/hard limit.
    """
    # 위상 정렬된 노드들을 순서대로 담아 반환할 결과 리스트입니다.
    topo = []
    # 이미 방문한 노드들의 메모리 주소(id)를 기록하여 중복 방문 및 사이클을 방지하는 집합입니다.
    visited = set()
    # 반복적 DFS를 위한 스택으로, 요소는 (현재 노드, 처리할 부모의 인덱스) 쌍을 담고 있습니다.
    stack = [(root, 0)]
    # 초기 루트 노드의 id를 방문 집합에 추가하여 방문 처리합니다.
    visited.add(id(root))
    # 노드 수가 경고 임계값을 넘었을 때 한 번만 경고 메시지를 출력하도록 상태를 기록하는 플래그 변수입니다.
    _warned = False

    # 스택이 비어있지 않은 동안 DFS 탐색을 계속 진행합니다.
    while stack:
        # 스택의 최상단에서 현재 노드와 그 노드의 탐색 중인 부모 인덱스를 가져옵니다.
        node, idx = stack[-1]
        # 현재 노드의 역전파 부모 노드들을 가져오되, 없을 경우 빈 튜플을 기본값으로 반환합니다.
        grad_parents = getattr(node, '_grad_parents', ())
        
        if idx < len(grad_parents):
            # 아직 탐색할 부모 노드가 남아있다면, 현재 노드의 상태를 (idx + 1)로 갱신하여 스택을 업데이트합니다.
            stack[-1] = (node, idx + 1)
            # 다음으로 방문할 부모 노드 객체를 가져옵니다.
            p = grad_parents[idx]
            # 해당 부모 노드의 고유 메모리 주소값을 구합니다.
            pid = id(p)
            
            if pid not in visited:
                # 해당 부모 노드를 아직 방문하지 않았다면 방문 집합에 추가합니다.
                visited.add(pid)
                # 새로운 부모 노드 탐색을 시작하기 위해 스택에 추가(인덱스 0)합니다.
                stack.append((p, 0))

                # 현재까지 방문한 전체 노드 수를 계산하여 노드 제한을 확인합니다.
                node_count = len(visited)
                if _max_graph_nodes_hard_limit is not None and node_count > _max_graph_nodes_hard_limit:
                    # 방문한 노드 수가 강제 제한 임계치를 초과했다면 RuntimeError를 발생시킵니다.
                    raise RuntimeError(
                        f"Autograd graph exceeds hard limit of {_max_graph_nodes_hard_limit} nodes. "
                        f"Use set_max_graph_nodes(hard_limit=N) to adjust."
                    )
                if not _warned and node_count > _max_graph_nodes_warning:
                    # 노드 수가 경고 임계치를 넘었고 아직 경고하지 않았다면 warnings 모듈을 통해 경고를 발생시킵니다.
                    import warnings
                    warnings.warn(
                        f"[AMEVA] Autograd graph has {node_count} nodes (warning threshold: "
                        f"{_max_graph_nodes_warning}). This may consume excessive memory. "
                        f"Consider checkpointing or reducing computation depth.",
                        RuntimeWarning, stacklevel=2
                    )
                    # 중복 경고를 방지하기 위해 플래그를 True로 변경합니다.
                    _warned = True
        else:
            # 현재 노드의 모든 부모 노드들에 대한 탐색이 끝났다면 스택에서 꺼냅니다.
            stack.pop()
            # 탐색이 완료된 노드는 자식들의 의존성이 모두 해결되었으므로 결과 리스트(topo)에 추가합니다.
            topo.append(node)

    # 역방향으로 위상 정렬된 모든 텐서들의 리스트를 반환합니다.
    return topo

