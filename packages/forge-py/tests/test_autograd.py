import sys
import unittest
import numpy as np
import forge as at
from forge.autograd import no_grad

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestAutograd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        self.device = at.current_device()

    async def test_forward_backward_rigorous(self):
        size = 1024
        print(f"\n--- 💥 EXTREME LIMITS: Autograd Backprop ({size}x{size}) ---")
        import time
        # We will do a full training step forward and backward, 
        # and compare the gradients with numpy ground truth.
        X_np = np.random.randn(size, size).astype(np.float32)
        W_np = np.random.randn(size, size).astype(np.float32)
        # Avoid broadcasting in this test since AMEVA-Forge Add.backward 
        # doesn't implement axis summation yet. We test pure matrix-sized gradients.
        b_np = np.random.randn(size, size).astype(np.float32)

        X = at.tensor(X_np, device=self.device, requires_grad=False)
        W = at.tensor(W_np, device=self.device, requires_grad=True)
        b = at.tensor(b_np, device=self.device, requires_grad=True)

        t0 = time.time()
        # Forward
        Z1 = X @ W
        Z2 = at.add(Z1, b)
        A = at.relu(Z2)
        
        # Start backprop (Explicit gradient since A is not a scalar)
        grad_A_np = np.ones((size, size), dtype=np.float32)
        A.backward(at.tensor(grad_A_np, device=self.device))
        
        # Async wait for readbacks
        b_grad_np = await b.grad.numpy_async()
        W_grad_np = await W.grad.numpy_async()
        print(f"WebGPU Forward + Backward (1 Million Gradients): {time.time() - t0:.3f}s")
        
        Z1_np = np.matmul(X_np, W_np)
        Z2_np = Z1_np + b_np
        dZ2_np = grad_A_np * (Z2_np > 0)
        
        # db = dZ2_np since b is (size, size) in this test
        expected_db = dZ2_np
        # dW = X.T @ dZ2
        expected_dW = np.matmul(X_np.T, dZ2_np)
        
        np.testing.assert_allclose(b_grad_np, expected_db, rtol=1e-3, err_msg="Bias gradient mismatch")
        np.testing.assert_allclose(W_grad_np, expected_dW, rtol=1e-3, err_msg="Weight gradient mismatch")
        
    async def test_no_grad_context(self):
        W = at.tensor([[1.0, 2.0]], device=self.device, requires_grad=True)
        X = at.tensor([[0.5], [0.5]], device=self.device, requires_grad=False)
        
        with no_grad():
            Z = W @ X
            
        self.assertFalse(hasattr(Z, 'requires_grad') and Z.requires_grad)

if __name__ == '__main__':
    unittest.main()
