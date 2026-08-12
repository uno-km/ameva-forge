import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestOptimizers(unittest.TestCase):
    def test_sgd(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        y = x * x
        y.backward()
        opt.step()
        np.testing.assert_allclose(x.numpy(), [2.0 - 0.1 * 4.0])
        
    def test_sgd_momentum(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1, momentum=0.9)
        (x * x).backward()
        opt.step()
        opt.zero_grad()
        (x * x).backward()
        opt.step()
        self.assertTrue(x.numpy()[0] < 1.6) # Should move faster
        
    def test_adam(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.Adam([x], lr=0.1)
        (x * x).backward()
        opt.step()
        self.assertTrue(x.numpy()[0] < 2.0)
        
    def test_zero_grad(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        (x * x).backward()
        self.assertIsNotNone(x.grad)
        opt.zero_grad()
        self.assertIsNone(x.grad)
