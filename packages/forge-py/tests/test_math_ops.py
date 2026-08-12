import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import forge as at
from forge import Tensor

class TestMathOps(unittest.TestCase):
    def test_exp(self):
        t = at.tensor([0.0, 1.0])
        res = at.exp(t)
        np.testing.assert_allclose(res.numpy(), [1.0, np.exp(1.0)])
        
    def test_log(self):
        t = at.tensor([1.0, np.exp(1.0)])
        res = at.log(t)
        np.testing.assert_allclose(res.numpy(), [0.0, 1.0])
        
    def test_exp_method(self):
        t = at.tensor([0.0, 1.0])
        res = t.exp()
        np.testing.assert_allclose(res.numpy(), [1.0, np.exp(1.0)])
        
    def test_log_method(self):
        t = at.tensor([1.0, np.exp(1.0)])
        res = t.log()
        np.testing.assert_allclose(res.numpy(), [0.0, 1.0])
