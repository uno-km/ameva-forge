import sys
import unittest
import numpy as np
import ameva_tensor as at
from ameva_tensor.autograd import no_grad

@unittest.skipUnless(sys.platform == 'emscripten', 'Requires Pyodide/WebGPU environment')
class TestAutograd(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await at.init()
        self.device = at.current_device()

    async def test_forward_backward_rigorous(self):
        # We will do a full training step forward and backward, 
        # and compare the gradients with numpy ground truth.
        X_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        W_np = np.array([[0.1, 0.2], 
                         [0.3, 0.4], 
                         [0.5, 0.6]], dtype=np.float32)
        b_np = np.array([[0.1, -0.1]], dtype=np.float32)

        X = at.tensor(X_np, device=self.device, requires_grad=False)
        W = at.tensor(W_np, device=self.device, requires_grad=True)
        b = at.tensor(b_np, device=self.device, requires_grad=True)

        # Forward
        Z1 = X @ W
        Z2 = at.add(Z1, b)
        A = at.relu(Z2)
        
        # Start backprop (Implicit sum gradient of 1)
        A.backward()
        
        # Async wait for readbacks
        b_grad_np = await b.grad.numpy_async()
        W_grad_np = await W.grad.numpy_async()
        
        # Check gradients math
        # Z1 = X @ W = [[1*0.1 + 2*0.3 + 3*0.5, 1*0.2 + 2*0.4 + 3*0.6]] = [[2.2, 2.8]]
        # Z2 = Z1 + b = [[2.3, 2.7]]
        # A = ReLU(Z2) = [[2.3, 2.7]]
        # dA = 1
        # dZ2 = dA * (Z2 > 0) = [[1, 1]]
        # db = dZ2 = [[1, 1]]
        # dW = X.T @ dZ2 = [[1], [2], [3]] @ [[1, 1]] = [[1, 1], [2, 2], [3, 3]]
        
        np.testing.assert_allclose(b_grad_np, np.array([[1.0, 1.0]], dtype=np.float32), err_msg="Bias gradient mismatch")
        
        expected_dW = np.array([[1.0, 1.0], 
                                [2.0, 2.0], 
                                [3.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(W_grad_np, expected_dW, err_msg="Weight gradient mismatch")
        
    async def test_no_grad_context(self):
        W = at.tensor([[1.0, 2.0]], device=self.device, requires_grad=True)
        X = at.tensor([[0.5], [0.5]], device=self.device, requires_grad=False)
        
        with no_grad():
            Z = W @ X
            
        self.assertFalse(hasattr(Z, 'requires_grad') and Z.requires_grad)

if __name__ == '__main__':
    unittest.main()
