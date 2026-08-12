import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestActivationFunctions(unittest.TestCase):
    def test_relu(self):
        t = at.tensor([-1.0, 0.0, 1.0, 2.0])
        res = at.relu(t)
        np.testing.assert_allclose(res.numpy(), [0.0, 0.0, 1.0, 2.0])
        
    def test_sigmoid(self):
        t = at.tensor([0.0, 100.0, -100.0])
        res = at.sigmoid(t)
        np.testing.assert_allclose(res.numpy(), [0.5, 1.0, 0.0], atol=1e-5)
        
    def test_tanh(self):
        t = at.tensor([0.0, 100.0, -100.0])
        res = at.tanh(t)
        np.testing.assert_allclose(res.numpy(), [0.0, 1.0, -1.0], atol=1e-5)
        
    def test_relu_method(self):
        t = at.tensor([-2.0, 3.0])
        res = t.relu()
        np.testing.assert_allclose(res.numpy(), [0.0, 3.0])
