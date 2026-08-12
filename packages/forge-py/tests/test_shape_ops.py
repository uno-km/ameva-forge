import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestShapeOps(unittest.TestCase):
    def test_reshape(self):
        t = at.tensor([[1, 2, 3], [4, 5, 6]])
        res = t.reshape((3, 2))
        self.assertEqual(res.shape, (3, 2))
        np.testing.assert_allclose(res.numpy(), [[1, 2], [3, 4], [5, 6]])
        
    def test_view(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = t.view(4)
        self.assertEqual(res.shape, (4,))
        np.testing.assert_allclose(res.numpy(), [1, 2, 3, 4])
        
    def test_numel(self):
        t = at.tensor([[1, 2], [3, 4]])
        self.assertEqual(t.numel(), 4)
        
    def test_reshape_function(self):
        t = at.tensor([1, 2, 3, 4])
        res = at.reshape(t, (2, 2))
        self.assertEqual(res.shape, (2, 2))
