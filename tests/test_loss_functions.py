import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestLossFunctions(unittest.TestCase):
    def test_mse_loss(self):
        pred = at.tensor([1.0, 2.0])
        target = at.tensor([1.0, 3.0])
        loss = at.F.mse_loss(pred, target)
        self.assertAlmostEqual(loss.numpy().item(), 0.5)
        
    def test_softmax(self):
        x = at.tensor([[1.0, 2.0, 3.0]])
        res = at.F.softmax(x)
        self.assertAlmostEqual(np.sum(res.numpy()), 1.0)
        
    def test_log_softmax(self):
        x = at.tensor([[1.0, 2.0, 3.0]])
        res = at.F.log_softmax(x)
        np.testing.assert_allclose(res.numpy(), np.log(at.F.softmax(x).numpy()), atol=1e-5)
        
    def test_cross_entropy(self):
        pred = at.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = at.tensor([0, 1])
        loss = at.F.cross_entropy(pred, target)
        self.assertTrue(loss.numpy().item() < 0.01)
