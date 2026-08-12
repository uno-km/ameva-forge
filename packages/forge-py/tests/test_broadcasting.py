import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestBroadcasting(unittest.TestCase):
    def test_broadcast_scalar(self):
        t1 = at.tensor([[1, 2], [3, 4]])
        res = t1 + 10
        np.testing.assert_allclose(res.numpy(), [[11, 12], [13, 14]])
        
    def test_broadcast_1d(self):
        t1 = at.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = at.tensor([10, 20, 30])
        res = t1 + t2
        np.testing.assert_allclose(res.numpy(), [[11, 22, 33], [14, 25, 36]])
        
    def test_broadcast_2d(self):
        t1 = at.tensor([[1], [2], [3]])
        t2 = at.tensor([10, 20, 30])
        res = t1 + t2
        np.testing.assert_allclose(res.numpy(), [[11, 21, 31], [12, 22, 32], [13, 23, 33]])
        
    def test_broadcast_sub(self):
        t1 = at.tensor([[10, 20], [30, 40]])
        t2 = at.tensor([5, 5])
        res = t1 - t2
        np.testing.assert_allclose(res.numpy(), [[5, 15], [25, 35]])
        
    def test_broadcast_mul(self):
        t1 = at.tensor([[1, 2], [3, 4]])
        t2 = at.tensor([2, 3])
        res = t1 * t2
        np.testing.assert_allclose(res.numpy(), [[2, 6], [6, 12]])
