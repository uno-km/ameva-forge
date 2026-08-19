"""
test_inplace_version_lock.py - PyTorch Autograd In-place Mutation & Version Lock Tests
"""
import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import forge as at
from forge.ops import tensor


class TestInplaceVersionLock(unittest.TestCase):
    def test_inplace_add_version_increment(self):
        x = tensor([1.0, 2.0, 3.0])
        self.assertEqual(x._version, 0)
        x.add_(5.0)
        self.assertEqual(x._version, 1)
        np.testing.assert_allclose(x.numpy(), [6.0, 7.0, 8.0])

    def test_inplace_mul_version_increment(self):
        x = tensor([2.0, 4.0])
        self.assertEqual(x._version, 0)
        x *= 3.0
        self.assertEqual(x._version, 1)
        np.testing.assert_allclose(x.numpy(), [6.0, 12.0])

    def test_saved_tensor_inplace_mutation_error(self):
        """
        Tests that mutating a tensor saved for backward triggers the exact PyTorch RuntimeError.
        y = x * x (x is saved in MulFunction)
        x.add_(1.0) -> x._version changes
        y.backward() -> throws RuntimeError
        """
        x = tensor([2.0, 3.0], requires_grad=True)
        y = x * x
        self.assertEqual(x._version, 0)

        # Mutate x in-place after forward
        x.add_(10.0)
        self.assertEqual(x._version, 1)

        # Backward must fail fast with PyTorch standard error
        with self.assertRaises(RuntimeError) as ctx:
            y.backward(tensor([1.0, 1.0]))

        self.assertIn("modified by an inplace operation", str(ctx.exception))

    def test_unmutated_backward_succeeds(self):
        """Tests that when no in-place mutation occurs, backward succeeds cleanly."""
        x = tensor([2.0, 3.0], requires_grad=True)
        y = x * x
        y.backward(tensor([1.0, 1.0]))
        np.testing.assert_allclose(x.grad.numpy(), [4.0, 6.0])

    def test_gpu_inplace_acyclic_dag(self):
        """Tests that GPU in-place operations create a clean, acyclic DAG without self-loops."""
        from forge.graph import GraphBuilder
        x = tensor([1.0, 2.0, 3.0], device="gpu")
        x.add_(5.0)
        x.mul_(2.0)
        
        self.assertEqual(x._version, 2)
        self.assertEqual(x.device, "gpu")
        
        # Verify GraphBuilder compiles without cycle error
        gb = GraphBuilder()
        gb.add_tensor(x)
        insts, inputs = gb.compile()
        self.assertTrue(len(insts) >= 3)
        self.assertIn("add", insts)
        self.assertIn("mul", insts)

    def test_gpu_inplace_gc_stress_and_handle_preservation(self):
        """Tests that forced GC does not prematurely destroy GPU handles during in-place mutations."""
        import gc
        import json
        from forge.graph import GraphBuilder

        x = tensor([1.0, 2.0, 3.0], device="gpu")
        x._handle_cell.handle = "realized_gpu_buffer_001"
        old_cell = x._handle_cell

        x.add_(5.0)
        
        # Force aggressive GC collection
        gc.collect()
        
        # Old cell must be preserved until graph compilation
        self.assertEqual(old_cell.handle, "realized_gpu_buffer_001")
        self.assertGreaterEqual(old_cell.ref_count, 1)

        # Graph compile must use the preserved handle in "load"
        gb = GraphBuilder()
        gb.add_tensor(x)
        insts_json, inputs = gb.compile()
        insts = json.loads(insts_json)
        self.assertEqual(insts[0]["op"], "load")
        self.assertEqual(insts[0]["handle"], "realized_gpu_buffer_001")

    def test_gpu_fill_and_zero_inplace(self):
        """Tests that fill_ and zero_ on GPU create clean AST and version increment."""
        import gc
        from forge.graph import GraphBuilder
        x = tensor([1.0, 2.0, 3.0, 4.0], device="gpu")
        x.fill_(7.0)
        self.assertEqual(x._version, 1)
        self.assertEqual(x.device, "gpu")

        # Force GC to verify stability
        gc.collect()

        gb = GraphBuilder()
        gb.add_tensor(x)
        insts, inputs = gb.compile()
        self.assertIn("fill", insts)

        x.zero_()
        self.assertEqual(x._version, 2)

    def test_move_tensor_state_clean_transfer(self):
        """Tests that _move_tensor_state transfers HandleCell without leaving zombie cells."""
        import gc
        from forge.functional import _move_tensor_state
        dst = tensor([1.0, 2.0], device="gpu")
        src = tensor([3.0, 4.0], device="gpu")
        src_cell = src._handle_cell

        _move_tensor_state(dst, src)

        # dst should now own src_cell
        self.assertIs(dst._handle_cell, src_cell)
        self.assertIsNone(src._handle_cell)

        # Force GC
        gc.collect()
        self.assertFalse(dst._disposed)

    def test_move_to_device_gc_preservation(self):
        """Tests that move_to_('gpu') maintains intact HandleCell after GC."""
        import gc
        p = tensor([1.0, 2.0, 3.0], device="cpu")
        p.move_to_("gpu")
        p._handle_cell.handle = "p_gpu_handle_test"

        gc.collect()
        self.assertEqual(p.device, "gpu")
        self.assertEqual(p._handle_cell.handle, "p_gpu_handle_test")
        self.assertGreaterEqual(p._handle_cell.ref_count, 1)

    def test_graph_builder_common_subexpression_elimination(self):
        """Tests that GraphBuilder eliminates redundant uploads for shared DAG inputs."""
        import json
        from forge.graph import GraphBuilder
        x = tensor([1.0, 2.0], device="gpu")
        y1 = x + 1.0
        y2 = x * 2.0

        gb = GraphBuilder()
        id1 = gb.add_tensor(y1)
        id2 = gb.add_tensor(y2)

        insts_json, inputs = gb.compile()
        insts = json.loads(insts_json)

        # Total nodes: upload(x), upload(1.0), add, upload(2.0), mul = 5
        self.assertEqual(len(insts), 5)
        # Inputs: x, 1.0, 2.0 = 3
        self.assertEqual(len(inputs), 3)

    def test_tensor_clone_and_autograd(self):
        """Tests that Tensor.clone() creates an independent tensor and propagates gradients."""
        x = tensor([2.0, 3.0], requires_grad=True)
        y = x.clone()
        z = y * 4.0
        z.backward(tensor([1.0, 1.0]))
        np.testing.assert_allclose(x.grad.numpy(), [4.0, 4.0])

    def test_tensor_pow_and_zero_safe_backward(self):
        """Tests that x ** p computes accurately and does not explode to NaN at x=0."""
        # 1. Standard power
        x = tensor([2.0, 3.0], requires_grad=True)
        y = x ** 2.0
        np.testing.assert_allclose(y.numpy(), [4.0, 9.0])
        y.backward(tensor([1.0, 1.0]))
        np.testing.assert_allclose(x.grad.numpy(), [4.0, 6.0])

        # 2. Square root / p=0.5 with x=0.0 (Zero-Safe Gradient Clamping)
        x_zero = tensor([0.0, 4.0], requires_grad=True)
        y_sqrt = x_zero ** 0.5
        np.testing.assert_allclose(y_sqrt.numpy(), [0.0, 2.0])
        y_sqrt.backward(tensor([1.0, 1.0]))
        self.assertFalse(np.isnan(x_zero.grad.numpy()).any())
        self.assertEqual(x_zero.grad.numpy()[0], 0.0) # Masked 0.0
        self.assertAlmostEqual(x_zero.grad.numpy()[1], 0.25) # 0.5 * 4^(-0.5) = 0.25

    def test_cross_entropy_soft_targets(self):
        """Tests that functional.cross_entropy accurately handles 2D soft probability targets."""
        from forge.functional import cross_entropy
        preds = tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True)
        soft_targets = tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        loss = cross_entropy(preds, soft_targets)
        self.assertGreater(float(loss.numpy()), 0.0)

        loss.backward()
        self.assertIsNotNone(preds.grad)
        self.assertEqual(preds.grad.shape, (2, 3))


if __name__ == '__main__':
    unittest.main()

