import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestReductionOps(unittest.TestCase):
    def test_sum_all(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = t.sum()
        self.assertEqual(res.shape, ())
        self.assertAlmostEqual(res.numpy().item(), 10.0)
        
    def test_mean_all(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = t.mean()
        self.assertEqual(res.shape, ())
        self.assertAlmostEqual(res.numpy().item(), 2.5)
        
    def test_sum_axis(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = at.sum_axis(t, axis=0)
        np.testing.assert_allclose(res.numpy(), [4, 6])
        
    def test_sum_axis_1(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = at.sum_axis(t, axis=1)
        np.testing.assert_allclose(res.numpy(), [3, 7])
