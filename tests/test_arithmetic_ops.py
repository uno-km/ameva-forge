import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestArithmeticOps(unittest.TestCase):
    def test_add(self):
        t1 = at.tensor([1.0, 2.0])
        t2 = at.tensor([3.0, 4.0])
        res = t1 + t2
        np.testing.assert_allclose(res.numpy(), [4.0, 6.0])

    def test_sub(self):
        t1 = at.tensor([5.0, 6.0])
        t2 = at.tensor([3.0, 2.0])
        res = t1 - t2
        np.testing.assert_allclose(res.numpy(), [2.0, 4.0])
        
    def test_mul(self):
        t1 = at.tensor([2.0, 3.0])
        t2 = at.tensor([4.0, 5.0])
        res = t1 * t2
        np.testing.assert_allclose(res.numpy(), [8.0, 15.0])
        
    def test_div(self):
        t1 = at.tensor([10.0, 15.0])
        t2 = at.tensor([2.0, 3.0])
        res = t1 / t2
        np.testing.assert_allclose(res.numpy(), [5.0, 5.0])
        
    def test_neg(self):
        t1 = at.tensor([1.0, -2.0])
        res = -t1
        np.testing.assert_allclose(res.numpy(), [-1.0, 2.0])

    def test_add_scalar(self):
        t = at.tensor([1.0, 2.0])
        res = t + 2.0
        np.testing.assert_allclose(res.numpy(), [3.0, 4.0])
