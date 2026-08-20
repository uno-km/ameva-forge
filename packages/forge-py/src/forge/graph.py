"""
graph.py - WebGPU 연산 그래프 빌더

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
"""
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .errors import AMEVAForgeSecurityError, AMEVAForgeValidationError

# WHAT: WebGPU 그래프 실행기가 허용하는 등록된 모든 정규 텐서 오퍼레이션의 화이트리스트
# WHY: 미등록된 함수명, dunder 메서드, 프로토타입 오염 공격을 원천 차단하기 위함
ALLOWED_OPS_WHITELIST = frozenset([
    'upload', 'load', 'matmul', 'matmul_tiled', 'batched_matmul', 'relu', 'add', 'mul', 'transpose', 'relu_backward',
    'sub', 'neg', 'div', 'exp', 'log', 'sigmoid', 'tanh', 'sigmoid_backward', 'tanh_backward',
    'fill', 'sum', 'max', 'sum_axis', 'max_axis', 'max_axis_backward', 'axpy', 'cat', 'where', 'pad', 'gather', 'scatter',
    'maxpool2d', 'avgpool2d', 'im2col', 'col2im', 'dropout', 'permute', 'matmul_bias_relu', 'reshape', 'slice', 'slice_backward',
    'reduce_axes', 'flash_attention', 'rope', 'rmsnorm', 'swiglu', 'unpack_quant', 'embedding', 'embedding_backward',
    'adam_step', 'sgd_momentum_step', 'sparse_cross_entropy', 'sparse_cross_entropy_backward', 'mse_loss', 'mse_loss_backward'
])


class GraphBuilder:
    """
    무엇을: 연산 노드들을 순서대로 기록하여 JSON 형태의 연산 그래프로 빌드해주는 클래스이다.
    왜: WebGPU 브릿지(JS)로 전송할 때 복잡한 텐서 연산을 하나의 배치(배치 연산)로 최적화하여 넘기기 위함이다.
    어떻게: 각 노드를 딕셔너리 형태로 nodes 리스트에 추가하고, 고유한 ID를 부여하여 그래프로 연결한다.
    """
    def __init__(self):
        """
        무엇을: GraphBuilder 인스턴스를 초기화한다.
        왜: 그래프를 담을 리스트와 노드 ID 발급 상태를 리셋하기 위함이다.
        어떻게: 빈 노드 리스트(nodes), 입력 리스트(inputs), 그리고 시작 ID를 1로 설정한다.
        """
        # 무엇을: 그래프를 구성하는 연산 노드들의 정보를 담은 딕셔너리 리스트.
        # 왜: 나중에 JSON으로 직렬화하기 위함이다.
        # 어떻게: add_op 등의 메서드가 호출될 때마다 새로운 노드가 추가된다.
        self.nodes: List[Dict[str, Any]] = []
        
        # 무엇을: 연산 그래프에서 사용하는 초기 입력 데이터 리스트.
        # 왜: 업로드 시 전달할 실제 데이터를 저장하기 위함이다.
        # 어떻게: add_upload를 통해 외부 데이터가 주입될 때마다 순서대로 추가된다.
        self.inputs: List[Any] = []
        
        # 무엇을: 다음 노드에 부여할 고유 식별자 번호.
        # 왜: 그래프 내 각 노드 간의 의존성을 ID 기반으로 연결하기 위함이다.
        # 어떻게: alloc_id 호출 시마다 1씩 증가한다.
        self.next_id = 1
        self.node_id_map: Dict[Tuple[int, int], int] = {}

    def alloc_id(self) -> int:
        """
        무엇을: 새로운 노드용 고유 ID를 할당하여 반환한다.
        왜: 연산 그래프 내 노드 간 입출력 관계(dependency)를 추적하기 위한 식별자가 필요하기 때문이다.
        어떻게: 현재 next_id 값을 리턴하고, next_id 자체는 1 증가시킨다.
        """
        idx = self.next_id
        self.next_id += 1
        return idx

    def add_upload(self, shape: Tuple[int, ...], data: Any = None) -> int:
        """
        무엇을: 외부 데이터를 GPU 메모리로 업로드하는 'upload' 노드를 그래프에 추가한다.
        왜: CPU 데이터를 초기 텐서 노드로 변환하기 위함이다.
        어떻게: 새 ID를 받고, shape의 각 차원을 int로 변환해 JSON 직렬화 가능하게 만들며, 데이터가 있으면 inputs 리스트에 추가한다.
        """
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Shape must be a tuple or list of ints, got {type(shape)}")
        clean_shape = []
        for d in shape:
            if not isinstance(d, (int, np.integer)) or int(d) < 0:
                raise ValueError(f"Shape dimensions must be non-negative integers, got {d}")
            clean_shape.append(int(d))
            
        if len(clean_shape) > 8:
            raise ValueError(f"Shape dimensions cannot exceed 8, got {len(clean_shape)}")

        node_id = self.alloc_id()
        self.nodes.append({"op": "upload", "id": node_id, "shape": clean_shape})
        if data is not None:
            self.inputs.append(data)
        return node_id

    def add_load(self, shape: Tuple[int, ...], handle: str) -> int:
        """
        무엇을: 이미 GPU에 존재하는 데이터를 로드하는 'load' 노드를 추가한다.
        왜: 기존 캐싱된 메모리나 다른 파이프라인에서 생성된 리소스를 그래프 안으로 끌어오기 위함이다.
        어떻게: 새 ID를 발급받고 노드 정보(핸들 포함)를 기록한 뒤 해당 ID를 리턴한다.
        """
        if not isinstance(handle, str) or not handle.strip():
            raise ValueError(f"Handle must be a non-empty string, got {handle}")
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Shape must be a tuple or list of ints, got {type(shape)}")
        clean_shape = [int(d) for d in shape]
        
        node_id = self.alloc_id()
        self.nodes.append({
            "op": "load",
            "id": node_id,
            "shape": clean_shape,
            "handle": handle
        })
        return node_id

    def add_op(
        self,
        op_name: str,
        shape: Tuple[int, ...],
        in_ids: List[int],
        params: Optional[List[Any]] = None
    ) -> int:
        """
        무엇을: 일반적인 텐서 연산(op) 노드를 그래프에 추가한다.
        왜: 사용자가 호출한 수학적 연산(add, mul 등)을 그래프에 기록하여 연산 순서를 정하기 위함이다.
        어떻게: 입력 노드 ID 리스트(in_ids)를 의존성으로 기록하고, 추가 파라미터가 있다면 JSON 포맷에 맞게 형변환하여 딕셔너리에 추가한다.
        """
        if not isinstance(op_name, str) or not op_name.strip():
            raise AMEVAForgeValidationError(f"[AMEVA-Forge Error] op_name must be a valid non-empty string, got: {op_name}")
        
        if op_name not in ALLOWED_OPS_WHITELIST:
            raise AMEVAForgeSecurityError(
                f"[AMEVA-Forge Security Alert] Prohibited, unrecognized, or unregistered op '{op_name}' detected in computation graph. "
                f"Execution halted to protect runtime integrity."
            )
            
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Shape must be a tuple or list of ints, got {type(shape)}")
        clean_shape = []
        for d in shape:
            if not isinstance(d, (int, np.integer)) or int(d) < 0:
                raise ValueError(f"Shape dimensions must be non-negative integers, got {d}")
            clean_shape.append(int(d))
            
        if len(clean_shape) > 8:
            raise ValueError(f"Shape dimensions cannot exceed 8, got {len(clean_shape)}")

        clean_in_ids = []
        for i_id in in_ids:
            if not isinstance(i_id, (int, np.integer)) or int(i_id) <= 0:
                raise ValueError(f"in_ids must contain positive integers, got {i_id}")
            clean_in_ids.append(int(i_id))

        node_id = self.alloc_id()
        node: Dict[str, Any] = {
            "op": op_name,
            "id": node_id,
            "shape": clean_shape,
            "in": clean_in_ids
        }
        if params is not None:
            clean_params = []
            for p in params:
                if isinstance(p, bool):
                    clean_params.append(p)
                elif isinstance(p, (float, np.floating)):
                    clean_params.append(float(p))
                elif isinstance(p, (int, np.integer)):
                    clean_params.append(int(p))
                else:
                    clean_params.append(p)
            node["params"] = clean_params
        self.nodes.append(node)
        return node_id

    def add_tensor(self, root: Any) -> int:
        """
        무엇을: Lazy Tensor의 DAG를 탐색하여 위상 정렬 순서대로 GraphBuilder에 노드를 추가한다.
        왜: 파이썬 텐서 객체로부터 WebGPU 연산 그래프를 한 번에 빌드하며, 공통 부분식(CSE)을 재사용하기 위함이다.
        어떻게: build_lazy_topo를 호출하고 (id, version) 키를 통해 중복 노드 생성을 방지한다.
        """
        from .tensor import build_lazy_topo
        topo = build_lazy_topo(root)
        for v in topo:
            key = (id(v), getattr(v, '_version', 0))
            if key in self.node_id_map:
                continue

            if getattr(v, '_handle', None) is not None:
                self.node_id_map[key] = self.add_load(v.shape, v._handle)
            elif getattr(v, '_lazy_op', None) == 'upload' or getattr(v, '_lazy_op', None) is None:
                data = getattr(v, '_data', None)
                self.node_id_map[key] = self.add_upload(v.shape, data)
            else:
                in_ids = [self.node_id_map[(id(p), getattr(p, '_version', 0))] for p in getattr(v, '_parents', ())]
                self.node_id_map[key] = self.add_op(v._lazy_op, v.shape, in_ids, getattr(v, '_lazy_params', None))
        return self.node_id_map[(id(root), getattr(root, '_version', 0))]

    def to_dict(self) -> Dict[str, Any]:
        """
        무엇을: 그래프 노드와 입력 데이터를 딕셔너리로 반환한다.
        """
        return {"instructions": self.nodes, "inputs": self.inputs}

    def compile(self) -> Tuple[str, List[Any]]:
        """
        무엇을: 현재까지 모인 노드 정보를 JSON 문자열과 입력 데이터 리스트로 묶어 반환한다.
        왜: 생성된 그래프를 JS 브릿지(WebGPU 백엔드) 측에 통신 가능한 형태로 전달하기 위함이다.
        어떻게: 내장 json 모듈을 이용해 nodes 리스트를 직렬화(dumps)하고, 입력 데이터(inputs)와 함께 튜플로 리턴한다.
        """
        return json.dumps(self.nodes), self.inputs

