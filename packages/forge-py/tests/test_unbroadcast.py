import unittest
import numpy as np
import pytest

from forge import tensor, randn
from forge.ops import _unbroadcast, add

class TestUnbroadcast(unittest.TestCase):
    def test_unbroadcast_cpu_gpu(self):
        devices = ['cpu']
        # If GPU is available (assuming testing environment allows), we'd test both.
        # However, to be safe, we'll try to initialize device first.
        try:
            from forge.device import init, is_available
            import asyncio
            asyncio.run(init())
            if is_available():
                devices.append('gpu')
        except Exception:
            pass
            
        for device in devices:
            # Test 1: (32, 128) -> (128,)
            grad1 = randn((32, 128), device=device)
            unb1 = _unbroadcast(grad1, (128,))
            self.assertEqual(unb1.shape, (128,))

            # Test 2: (64, 32, 128) -> (128,)
            grad2 = randn((64, 32, 128), device=device)
            unb2 = _unbroadcast(grad2, (128,))
            self.assertEqual(unb2.shape, (128,))

            # Test 3: (3, 4, 5) -> (1, 4, 1)
            grad3 = randn((3, 4, 5), device=device)
            unb3 = _unbroadcast(grad3, (1, 4, 1))
            self.assertEqual(unb3.shape, (1, 4, 1))

            # Test 4: Same shape bypass (32, 32) -> (32, 32)
            grad4 = randn((32, 32), device=device)
            unb4 = _unbroadcast(grad4, (32, 32))
            self.assertEqual(unb4.shape, (32, 32))

            # Test 5: Scalar broadcast backward (32, 128) -> ()
            grad5 = randn((32, 128), device=device)
            unb5 = _unbroadcast(grad5, ())
            self.assertEqual(unb5.shape, ())

    def test_end_to_end_broadcast_backward(self):
        # Additional check to ensure backward propagates correctly
        a = randn((32, 128), requires_grad=True)
        b = randn((128,), requires_grad=True)
        c = add(a, b)
        loss = c.sum()
        loss.backward()
        
        self.assertEqual(a.grad.shape, (32, 128))
        self.assertEqual(b.grad.shape, (128,))

if __name__ == '__main__':
    unittest.main()
