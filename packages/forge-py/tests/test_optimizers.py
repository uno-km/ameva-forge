import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import forge as at
from forge import Tensor

class TestOptimizers(unittest.TestCase):
    def test_sgd(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        y = x * x
        y.backward()
        opt.step()
        np.testing.assert_allclose(x.numpy(), [2.0 - 0.1 * 4.0])
        
    def test_sgd_momentum(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1, momentum=0.9)
        (x * x).backward()
        opt.step()
        opt.zero_grad()
        (x * x).backward()
        opt.step()
        self.assertTrue(x.numpy()[0] < 1.6) # Should move faster
        
    def test_adam(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.Adam([x], lr=0.1)
        (x * x).backward()
        opt.step()
        self.assertTrue(x.numpy()[0] < 2.0)
        
    def test_zero_grad(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        (x * x).backward()
        self.assertIsNotNone(x.grad)
        opt.zero_grad()
        self.assertIsNone(x.grad)

    def test_sgd_step_async_cpu_parameter(self):
        import asyncio
        x = at.tensor([2.0], device="cpu", requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        (x * x).backward()
        asyncio.run(opt.step_async())
        np.testing.assert_allclose(x.numpy(), [1.6], atol=1e-5)

    def test_adam_gpu_raises_device_error(self):
        x = at.tensor([2.0], device="gpu", requires_grad=True)
        x.grad = at.tensor([4.0], device="gpu")
        opt = at.optim.Adam([x], lr=0.1)
        with self.assertRaises(at.AMEVAForgeDeviceError):
            opt.step()

    def test_clip_grad_norm_cpu_and_gpu(self):
        # CPU normal clipping
        x = at.tensor([2.0, 3.0], device="cpu", requires_grad=True)
        x.grad = at.tensor([3.0, 4.0], device="cpu") # norm = 5.0
        at.optim.clip_grad_norm([x], max_norm=2.5)
        np.testing.assert_allclose(x.grad.numpy(), [1.5, 2.0], atol=1e-5)

        # GPU raises explicit device error
        x_gpu = at.tensor([2.0], device="gpu", requires_grad=True)
        x_gpu.grad = at.tensor([4.0], device="gpu")
        with self.assertRaises(at.AMEVAForgeDeviceError):
            at.optim.clip_grad_norm([x_gpu], max_norm=1.0)

    def test_clip_grad_value_cpu_and_gpu(self):
        # CPU normal value clipping
        x = at.tensor([2.0, 3.0], device="cpu", requires_grad=True)
        x.grad = at.tensor([-5.0, 5.0], device="cpu")
        at.optim.clip_grad_value([x], clip_value=2.0)
        np.testing.assert_allclose(x.grad.numpy(), [-2.0, 2.0], atol=1e-5)

        # GPU raises explicit device error
        x_gpu = at.tensor([2.0], device="gpu", requires_grad=True)
        x_gpu.grad = at.tensor([4.0], device="gpu")
        with self.assertRaises(at.AMEVAForgeDeviceError):
            at.optim.clip_grad_value([x_gpu], clip_value=1.0)

    def test_sgd_step_async_gpu_grad_disposal(self):
        import asyncio
        import json
        import forge.bridge as bridge
        
        orig_execute = bridge.js_execute_graph
        compiled_nodes = []
        try:
            async def intercept_execute(instructions, inputs):
                inst_list = json.loads(instructions) if isinstance(instructions, str) else instructions
                compiled_nodes.extend(inst_list)
                res = {}
                for inst in inst_list:
                    if inst.get("op") == "axpy":
                        res[str(inst["id"])] = "handle_param"
                return res
                
            bridge.js_execute_graph = intercept_execute
            
            p = at.Tensor(shape=(2,), dtype='float32', device='gpu', handle='handle_param', requires_grad=True)
            grad_tensor = at.Tensor(shape=(2,), dtype='float32', device='gpu', handle='handle_grad')
            p.grad = grad_tensor
            
            opt = at.optim.SGD([p], lr=0.1)
            asyncio.run(opt.step_async())
            
            # Verify compiled graph structure
            axpy_nodes = [n for n in compiled_nodes if n.get("op") == "axpy"]
            self.assertEqual(len(axpy_nodes), 1)
            self.assertEqual(axpy_nodes[0]["params"], [2, 0.1])
            
            # Verify gradient disposal and param grad clearing
            self.assertIsNone(p.grad)
            self.assertTrue(grad_tensor._disposed)
        finally:
            bridge.js_execute_graph = orig_execute

if __name__ == '__main__':
    unittest.main()


