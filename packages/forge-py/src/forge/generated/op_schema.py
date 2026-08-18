"""
AUTO-GENERATED FILE. DO NOT MODIFY MANUALLY.
Generated from packages/forge/schema/release1-ops.json
Run `py -3 scripts/generate_release1_contracts.py` to regenerate.
"""

from typing import Dict, Any, List

RELEASE1_OP_SCHEMA: Dict[str, Dict[str, Any]] = {'add': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'sub': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'mul': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'div': {'inputs': 2, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'neg': {'inputs': 1, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'matmul': {'inputs': 2, 'params': [{'name': 'M', 'type': 'positive-int'}, {'name': 'N', 'type': 'positive-int'}, {'name': 'K', 'type': 'positive-int'}], 'output': 'matmul-2d', 'dtypes': ['float32']}, 'transpose': {'inputs': 1, 'params': [{'name': 'M', 'type': 'positive-int'}, {'name': 'N', 'type': 'positive-int'}], 'output': 'transpose-2d', 'dtypes': ['float32']}, 'reshape': {'inputs': 1, 'params': [{'name': 'targetShape', 'type': 'shape-tuple'}], 'output': 'reshape', 'dtypes': ['float32']}, 'sum': {'inputs': 1, 'params': [], 'output': 'scalar', 'dtypes': ['float32']}, 'relu': {'inputs': 1, 'params': [], 'output': 'same-shape', 'dtypes': ['float32']}, 'relu_backward': {'inputs': 2, 'params': [{'name': 'numElements', 'type': 'positive-int'}], 'output': 'same-shape', 'dtypes': ['float32']}, 'mse_loss': {'inputs': 2, 'params': [], 'output': 'scalar', 'dtypes': ['float32']}, 'mse_loss_backward': {'inputs': 2, 'params': [{'name': 'numElements', 'type': 'positive-int'}], 'output': 'same-shape', 'dtypes': ['float32']}, 'axpy': {'inputs': 2, 'params': [{'name': 'numElements', 'type': 'positive-int'}, {'name': 'alpha', 'type': 'float32'}], 'output': 'alias-input-0', 'dtypes': ['float32']}}

RELEASE1_OPS: List[str] = list(RELEASE1_OP_SCHEMA.keys())
