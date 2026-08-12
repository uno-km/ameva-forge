from typing import List
from .tensor import Tensor
from .ops import random, zeros, transpose, relu, matmul
import numpy as np

class Module:
    def __init__(self):
        # VUL-011 Fix: object.__setattr__로 안전하게 초기화
        # __setattr__ 오버라이드 전에 내부 딕셔너리를 먼저 생성
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_params', {})
    
    def forward(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def parameters(self) -> List[Tensor]:
        params = list(self._params.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Module):
            self._modules[name] = value
        if isinstance(value, Tensor) and getattr(value, 'requires_grad', False):
            self._params[name] = value
        object.__setattr__(self, name, value)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Kaiming initialization
        scale = (2.0 / in_features) ** 0.5
        w_data = np.random.randn(out_features, in_features).astype(np.float32) * scale
        self.weight = Tensor(shape=(out_features, in_features), dtype='float32', device='cpu', data=w_data, requires_grad=True)
        
        if bias:
            self.bias = Tensor(shape=(out_features,), dtype='float32', device='cpu',
                             data=np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x):
        out = matmul(x, transpose(self.weight))
        if self.bias is not None:
            from .ops import add
            out = add(out, self.bias)  # broadcasting: (batch, out) + (out,)
        return out


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Sigmoid(Module):
    def forward(self, x):
        from .ops import sigmoid
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        from .ops import tanh_op
        return tanh_op(x)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
