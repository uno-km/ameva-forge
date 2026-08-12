from typing import List
from .tensor import Tensor
import numpy as np

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [None] * len(self.params)
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad_data = p.grad._data if hasattr(p.grad, '_data') and p.grad._data is not None else p.grad.numpy()
            param_data = p._data if hasattr(p, '_data') and p._data is not None else p.numpy()
            
            if self.momentum > 0:
                if self.velocity[i] is None:
                    self.velocity[i] = grad_data.copy()
                else:
                    self.velocity[i] = self.momentum * self.velocity[i] + grad_data
                param_data = param_data - self.lr * self.velocity[i]
            else:
                param_data = param_data - self.lr * grad_data
            
            p._data = param_data.astype(np.float32)
            p.grad = None


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad._data if hasattr(p.grad, '_data') and p.grad._data is not None else p.grad.numpy()
            param_data = p._data if hasattr(p, '_data') and p._data is not None else p.numpy()
            
            if self.m[i] is None:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param_data = param_data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p._data = param_data.astype(np.float32)
            p.grad = None
