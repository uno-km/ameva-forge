import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestAutograd(unittest.TestCase):
    def test_add_backward(self):
        x = at.tensor([2.0], requires_grad=True)
        y = x + 3.0
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0])
        
    def test_mul_backward(self):
        x = at.tensor([2.0], requires_grad=True)
        y = x * 3.0
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [3.0])
        
    def test_matmul_backward(self):
        w = at.tensor([[1.0, 2.0]], requires_grad=True)
        x = at.tensor([[3.0], [4.0]])
        y = (w @ x).sum()
        y.backward()
        np.testing.assert_allclose(w.grad.numpy(), [[3.0, 4.0]])
        
    def test_relu_backward(self):
        x = at.tensor([-2.0, 2.0], requires_grad=True)
        y = x.relu().sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0])
        
    def test_broadcast_backward(self):
        x = at.tensor([[1.0, 2.0]], requires_grad=True) # (1, 2)
        y = at.tensor([[3.0], [4.0]]) # (2, 1)
        z = (x + y).sum()
        z.backward()
        np.testing.assert_allclose(x.grad.numpy(), [[2.0, 2.0]])
