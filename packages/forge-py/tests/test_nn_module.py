import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import forge as at
from forge import Tensor

class TestNNModule(unittest.TestCase):
    def test_linear(self):
        layer = at.nn.Linear(2, 3)
        x = at.tensor([[1.0, 2.0]])
        out = layer(x)
        self.assertEqual(out.shape, (1, 3))
        self.assertEqual(len(layer.parameters()), 2) # weight, bias
        
    def test_relu_module(self):
        layer = at.nn.ReLU()
        x = at.tensor([[-1.0, 2.0]])
        out = layer(x)
        np.testing.assert_allclose(out.numpy(), [[0.0, 2.0]])
        
    def test_sequential(self):
        model = at.nn.Sequential(
            at.nn.Linear(2, 4),
            at.nn.ReLU(),
            at.nn.Linear(4, 1)
        )
        x = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = model(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(len(model.parameters()), 4)
