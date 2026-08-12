import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestErrorHandling(unittest.TestCase):
    def test_shape_mismatch_add(self):
        t1 = at.ones((2, 2))
        t2 = at.ones((3, 3))
        with self.assertRaises(at.AMEVAForgeShapeError):
            t1 + t2
            
    def test_disposed_tensor(self):
        t = at.tensor([1, 2])
        t.dispose()
        with self.assertRaises(at.AMEVAForgeDisposedError):
            t + 1
            
    def test_backward_no_grad(self):
        t = at.tensor([1.0], requires_grad=False)
        with self.assertRaises(RuntimeError):
            t.backward()
            
    def test_invalid_shape(self):
        with self.assertRaises((at.AMEVAForgeShapeError, TypeError)):
            at.zeros("invalid")
