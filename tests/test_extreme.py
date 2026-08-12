import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestExtreme(unittest.TestCase):
    def test_large_tensor(self):
        t1 = at.ones((100, 100))
        t2 = at.ones((100, 100))
        res = t1 @ t2
        self.assertEqual(res.numpy()[0,0], 100.0)
        
    def test_deep_computation_chain(self):
        t = at.tensor([1.0], requires_grad=True)
        curr = t
        for _ in range(100):
            curr = curr * 1.01
        curr.sum().backward()
        self.assertAlmostEqual(t.grad.numpy()[0], 1.01 ** 100, places=2)
        
    def test_gradient_accumulation(self):
        x = at.tensor([2.0], requires_grad=True)
        for _ in range(50):
            y = x * x
            y.backward()
        self.assertEqual(x.grad.numpy()[0], 4.0 * 50)
        
    def test_xor_training(self):
        x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_data = [[0], [1], [1], [0]]
        
        model = at.nn.Sequential(
            at.nn.Linear(2, 4),
            at.nn.Tanh(),
            at.nn.Linear(4, 1),
            at.nn.Sigmoid()
        )
        opt = at.optim.Adam(model.parameters(), lr=0.1)
        
        x = at.tensor(x_data)
        y = at.tensor(y_data)
        
        for epoch in range(2000):
            opt.zero_grad()
            pred = model(x)
            loss = at.F.mse_loss(pred, y)
            if loss.numpy().item() < 0.05:
                break
            loss.backward()
            opt.step()
            
        self.assertTrue(loss.numpy().item() < 0.05)
