import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import forge as at
from forge import Tensor

class TestEdgeCases(unittest.TestCase):
    def test_nan(self):
        t = at.tensor([float('nan'), 1.0])
        res = t + 1
        self.assertTrue(np.isnan(res.numpy()[0]))
        
    def test_inf(self):
        t = at.tensor([float('inf'), 1.0])
        res = t * 2
        self.assertTrue(np.isinf(res.numpy()[0]))
        
    def test_zero_dim(self):
        t = at.tensor(5.0)
        self.assertEqual(t.shape, ())
        res = t + 2.0
        self.assertEqual(res.shape, ())
        self.assertEqual(res.numpy().item(), 7.0)
        
    def test_empty_tensor(self):
        with self.assertRaises(at.AMEVAForgeShapeError):
            at.zeros((0, 5))
            
    def test_negative_zero(self):
        t = at.tensor([-0.0])
        self.assertEqual(t.numpy()[0], 0.0)
        
    def test_denormals(self):
        t = at.tensor([1e-40])
        res = t * 2
        self.assertTrue(res.numpy()[0] > 0.0)
        self.assertTrue(res.numpy()[0] < 1e-39)
