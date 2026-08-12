import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestMatrixOps(unittest.TestCase):
    def test_matmul_2d(self):
        t1 = at.tensor([[1, 2], [3, 4]])
        t2 = at.tensor([[5, 6], [7, 8]])
        res = t1 @ t2
        np.testing.assert_allclose(res.numpy(), [[19, 22], [43, 50]])
        
    def test_transpose_2d(self):
        t1 = at.tensor([[1, 2, 3], [4, 5, 6]])
        res = at.transpose(t1)
        self.assertEqual(res.shape, (3, 2))
        np.testing.assert_allclose(res.numpy(), [[1, 4], [2, 5], [3, 6]])
        
    def test_matmul_shape_mismatch(self):
        t1 = at.tensor([[1, 2]])
        t2 = at.tensor([[1, 2, 3]])
        with self.assertRaises(at.AMEVAForgeShapeError):
            t1 @ t2

    def test_matmul_1d_not_supported(self):
        t1 = at.tensor([1, 2])
        t2 = at.tensor([3, 4])
        with self.assertRaises(at.AMEVAForgeShapeError):
            t1 @ t2
