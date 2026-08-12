import json
from typing import Any, Dict, List, Optional, Tuple


class GraphBuilder:
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.inputs: List[Any] = []
        self.next_id = 1

    def alloc_id(self) -> int:
        idx = self.next_id
        self.next_id += 1
        return idx

    def add_upload(self, shape: Tuple[int, ...], data: Any = None) -> int:
        node_id = self.alloc_id()
        # NM-01 Fix: shape dim을 int()로 변환하여 numpy.intp → JSON-safe Python int
        self.nodes.append({"op": "upload", "id": node_id, "shape": [int(d) for d in shape]})
        if data is not None:
            self.inputs.append(data)
        return node_id

    def add_load(self, shape: Tuple[int, ...], handle: str) -> int:
        node_id = self.alloc_id()
        # NM-01 Fix: shape dim int() 변환
        self.nodes.append({
            "op": "load",
            "id": node_id,
            "shape": [int(d) for d in shape],
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
        node_id = self.alloc_id()
        node: Dict[str, Any] = {
            "op": op_name,
            "id": node_id,
            # NM-01 Fix: shape dim int() 변환
            "shape": [int(d) for d in shape],
            "in": in_ids
        }
        if params is not None:
            # NM-01 Fix: params 내 numpy 타입 → Python int/float 변환
            node["params"] = [int(p) if hasattr(p, '__int__') else p for p in params]
        self.nodes.append(node)
        return node_id

    def compile(self) -> Tuple[str, List[Any]]:
        return json.dumps(self.nodes), self.inputs
