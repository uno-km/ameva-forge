import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import forge as at
from forge import Tensor

class TestTensorCreation(unittest.TestCase):
    def test_tensor_from_list(self):
        t = at.tensor([1, 2, 3])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.device, "cpu")
        np.testing.assert_array_equal(t.numpy(), np.array([1, 2, 3], dtype=np.float32))

    def test_tensor_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]])
        t = at.tensor(arr)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_array_equal(t.numpy(), arr)
        
    def test_zeros(self):
        t = at.zeros((2, 3))
        self.assertEqual(t.shape, (2, 3))
        np.testing.assert_array_equal(t.numpy(), np.zeros((2, 3)))
        
    def test_ones(self):
        t = at.ones((2, 3))
        self.assertEqual(t.shape, (2, 3))
        np.testing.assert_array_equal(t.numpy(), np.ones((2, 3)))
        
    def test_full(self):
        t = at.full((2, 2), 5.0)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_array_equal(t.numpy(), np.full((2, 2), 5.0))
        
    def test_random(self):
        t = at.random((3, 3))
        self.assertEqual(t.shape, (3, 3))
        self.assertTrue(np.all((t.numpy() >= 0) & (t.numpy() <= 1)))
        
    def test_requires_grad(self):
        t = at.tensor([1.0], requires_grad=True)
        self.assertTrue(t.requires_grad)
        t = at.tensor([1.0], requires_grad=False)
        self.assertFalse(t.requires_grad)

    def test_zeros_like(self):
        t1 = at.tensor([[1,2],[3,4]])
        t2 = at.zeros_like(t1)
        self.assertEqual(t2.shape, (2,2))
        np.testing.assert_array_equal(t2.numpy(), np.zeros((2,2)))
