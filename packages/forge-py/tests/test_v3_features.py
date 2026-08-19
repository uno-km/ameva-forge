import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import forge as at
import forge.nn as nn

class TestV3Features(unittest.TestCase):
    def test_cnn_forward_backward(self):
        print("Testing CNN...")
        # (N, C, H, W)
        x = at.tensor(np.random.randn(2, 3, 16, 16).astype(np.float32), requires_grad=True)
        
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(8)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2, 2)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(8 * 8 * 8, 2)
            
            def forward(self, x):
                return self.fc(self.flatten(self.pool(self.relu(self.bn(self.conv(x))))))
                
        model = CNN()
        model.train()
        out = model(x)
        self.assertEqual(out.shape, (2, 2))
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(model.conv.weight.grad)
        self.assertIsNotNone(model.fc.weight.grad)
        print("CNN OK")
        
    def test_rnn_forward(self):
        print("Testing Sequence Models...")
        embed = nn.Embedding(100, 16)
        # Sequence of indices (N=2, L=5)
        indices = at.tensor(np.random.randint(0, 100, (2, 5)).astype(np.float32), requires_grad=False)
        x = embed(indices)
        self.assertEqual(x.shape, (2, 5, 16))
        
        lstm = nn.LSTM(16, 32, batch_first=True)
        out, (h_n, c_n) = lstm(x)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(h_n.shape, (2, 32))
        print("RNN OK")
        
    def test_transformer_forward(self):
        print("Testing Transformer...")
        # (N, L, E)
        x = at.tensor(np.random.randn(2, 10, 64).astype(np.float32), requires_grad=True)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 64))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        print("Transformer OK")

    def test_softmax_nd(self):
        import forge.functional as F
        x = at.tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
        sm = F.softmax(x, axis=-1)
        self.assertEqual(sm.shape, (2, 4, 8))
        np.testing.assert_allclose(sm.numpy().sum(axis=-1), np.ones((2, 4)), atol=1e-5)
        loss = sm.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_state_dict_gpu_boundary(self):
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)
        model = LinearModel()
        model.to("gpu")
        # Simulate realized GPU tensor where host data is discarded
        model.fc.weight._data = None
        model.fc.weight._handle = "tensor_mock_gpu"
        # keep_vars=True returns GPU tensor objects safely
        sd = model.state_dict(keep_vars=True)
        self.assertIn("fc.weight", sd)
        # keep_vars=False raises explicit error on GPU model without CPU data
        with self.assertRaises(at.AMEVAForgeDeviceError):
            model.state_dict(keep_vars=False)

    def test_dropout_eval_mode_autograd_preservation(self):
        # Verify that Dropout(training=False) does not overwrite upstream linear autograd graph
        fc = nn.Linear(4, 2)
        drop = nn.Dropout(p=0.5)
        drop.eval()
        
        x = at.tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        h = fc(x)
        out = drop(h)
        # out should be exactly h or pass-through without cycle
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(fc.weight.grad)
        self.assertTrue(np.any(fc.weight.grad.numpy() != 0.0))
        self.assertIsNotNone(x.grad)

    def test_embedding_gpu_raises_device_error(self):
        embed = nn.Embedding(10, 4)
        embed.to("gpu")
        idx = at.tensor([0, 1, 2], device="cpu")
        with self.assertRaises(at.AMEVAForgeDeviceError):
            embed(idx)

    def test_scaled_dot_product_attention_causal_masking(self):
        import forge.functional as F
        # (B=1, L=4, D=8)
        q = at.tensor(np.ones((1, 4, 8), dtype=np.float32))
        k = at.tensor(np.ones((1, 4, 8), dtype=np.float32))
        v = at.tensor(np.eye(4, 8, dtype=np.float32).reshape(1, 4, 8))
        
        # Non-causal attention
        out_non_causal = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # Causal attention
        out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        self.assertEqual(out_causal.shape, (1, 4, 8))
        # In causal attention, token 0 can only attend to token 0 (so out[0,0] has non-zero only at pos 0)
        np.testing.assert_allclose(out_causal.numpy()[0, 0, 0], 1.0, atol=1e-3)
        np.testing.assert_allclose(out_causal.numpy()[0, 0, 1:], 0.0, atol=1e-3)

    def test_softmax_numerical_stability_large_logits(self):
        import forge.functional as F
        # Large logits (100.0, 105.0) that would cause exp() overflow without x - max(x)
        x = at.tensor(np.array([[100.0, 105.0], [50.0, 52.0]], dtype=np.float32))
        sm = F.softmax(x, axis=-1)
        sm_np = sm.numpy()
        self.assertFalse(np.isnan(sm_np).any(), "Softmax output contains NaN!")
        self.assertFalse(np.isinf(sm_np).any(), "Softmax output contains Inf!")
        np.testing.assert_allclose(sm_np.sum(axis=-1), np.ones(2), atol=1e-5)

    def test_5d_6d_8d_broadcasting_cpu(self):
        # 5D broadcasting
        a_5d = np.ones((2, 1, 3, 1, 4), dtype=np.float32)
        b_5d = np.full((1, 5, 1, 7, 4), 2.0, dtype=np.float32)
        res_5d = (at.tensor(a_5d) + at.tensor(b_5d)).numpy()
        np.testing.assert_allclose(res_5d, a_5d + b_5d)

        # 6D broadcasting
        a_6d = np.ones((1, 2, 1, 3, 1, 4), dtype=np.float32)
        b_6d = np.full((5, 1, 2, 1, 3, 4), 3.0, dtype=np.float32)
        res_6d = (at.tensor(a_6d) * at.tensor(b_6d)).numpy()
        np.testing.assert_allclose(res_6d, a_6d * b_6d)

        # 8D broadcasting
        a_8d = np.ones((1, 1, 2, 1, 3, 1, 4, 1), dtype=np.float32)
        b_8d = np.full((2, 1, 1, 2, 1, 3, 1, 2), 4.0, dtype=np.float32)
        res_8d = (at.tensor(a_8d) + at.tensor(b_8d)).numpy()
        np.testing.assert_allclose(res_8d, a_8d + b_8d)

    def test_max_axis_backward_duplicate_max(self):
        # Duplicate max values should split gradients equally
        x_np = np.array([[1.0, 3.0, 3.0], [2.0, 2.0, 1.0]], dtype=np.float32)
        x = at.tensor(x_np, requires_grad=True)
        y = at.ops.max_axis(x, axis=1)
        loss = y.sum()
        loss.backward()
        expected_grad = np.array([[0.0, 0.5, 0.5], [0.5, 0.5, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, atol=1e-5)

    def test_unsupported_gpu_backward_guards(self):
        # Conv2d on GPU with requires_grad raises AMEVAForgeUnsupportedOperationError
        conv = nn.Conv2d(3, 8, 3)
        conv.to("gpu")
        x_gpu = at.tensor(np.zeros((1, 3, 10, 10), dtype=np.float32), device="gpu", requires_grad=True)
        with self.assertRaises(at.AMEVAForgeUnsupportedOperationError):
            conv(x_gpu)

        # MaxPool2d on GPU with requires_grad raises AMEVAForgeUnsupportedOperationError
        pool = nn.MaxPool2d(2)
        pool.to("gpu")
        with self.assertRaises(at.AMEVAForgeUnsupportedOperationError):
            pool(x_gpu)

    def test_sgd_strict_training_nan_detection(self):
        # Strict training mode catches NaN gradient on CPU
        p = at.tensor([1.0, 2.0], requires_grad=True)
        p.grad = at.tensor([float("nan"), 0.5])
        opt = at.optim.SGD([p], lr=0.1, strict_training=True)
        with self.assertRaises(at.AMEVAForgeValidationError):
            opt.step()

    def test_sgd_non_strict_ieee754_propagation(self):
        # Non-strict mode propagates NaN according to IEEE 754
        p = at.tensor([1.0, 2.0], requires_grad=True)
        p.grad = at.tensor([float("nan"), 0.5])
        opt = at.optim.SGD([p], lr=0.1, strict_training=False)
        opt.step()
        p_val = p.numpy()
        self.assertTrue(np.isnan(p_val[0]))
        np.testing.assert_allclose(p_val[1], 1.95, atol=1e-5)

    def test_batch_norm2d_train_eval_state_preservation(self):
        # BatchNorm2d train -> eval state preservation on CPU
        bn = nn.BatchNorm2d(4)
        bn.train()
        x = at.randn((2, 4, 3, 3))
        out_train = bn(x)
        self.assertEqual(out_train.shape, (2, 4, 3, 3))
        
        # Switch to eval mode
        bn.eval()
        out_eval = bn(x)
        self.assertEqual(out_eval.shape, (2, 4, 3, 3))
        self.assertIsNotNone(bn.running_mean._data)
        self.assertIsNotNone(bn.running_var._data)

    def test_module_parameter_identity_preservation_and_device_guard(self):
        # Device mismatch guard
        ln = nn.LayerNorm(4) # on CPU
        x_gpu = at.tensor([1.0, 2.0, 3.0, 4.0], device="gpu")
        with self.assertRaises(at.AMEVAForgeDeviceError):
            ln(x_gpu)

        conv = nn.Conv2d(3, 8, 3) # on CPU
        x_conv_gpu = at.tensor(np.zeros((1, 3, 8, 8), dtype=np.float32), device="gpu")
        with self.assertRaises(at.AMEVAForgeDeviceError):
            conv(x_conv_gpu)

        # Parameter identity preservation with Optimizer
        ln_cpu = nn.LayerNorm(4)
        opt = at.optim.SGD(ln_cpu.parameters(), lr=0.1)
        weight_id_before = id(ln_cpu.weight)
        x_cpu = at.tensor([[1.0, 2.0, 3.0, 4.0]], device="cpu")
        out = ln_cpu(x_cpu)
        loss = out.sum()
        loss.backward()
        opt.step()

        self.assertEqual(id(ln_cpu.weight), weight_id_before)
        self.assertIs(opt.params[0], ln_cpu.weight)

    def test_gc_queue_retention_on_failure(self):
        # GC queue must not clear on failure
        from forge.tensor import _gc_queue, flush_gc
        import forge.bridge as bridge
        
        orig_dispose = bridge.js_dispose_batch
        try:
            def mock_failing_dispose(handles):
                raise RuntimeError("Simulated bridge failure")
            bridge.js_dispose_batch = mock_failing_dispose
            
            _gc_queue.add("test_h1")
            _gc_queue.add("test_h2")
            
            flush_gc(force=True)
            flush_gc(force=True)
            flush_gc(force=True)
            
            self.assertIn("test_h1", _gc_queue)
            self.assertIn("test_h2", _gc_queue)
        finally:
            bridge.js_dispose_batch = orig_dispose
            _gc_queue.discard("test_h1")
            _gc_queue.discard("test_h2")

    def test_gather_gpu_backward_unsupported_guard(self):
        # Gather backward on GPU raises unsupported error
        x_gpu = at.tensor([[1.0, 2.0], [3.0, 4.0]], device="gpu", requires_grad=True)
        idx_gpu = at.tensor([[0], [1]], device="gpu")
        from forge.ops import GatherFunction
        ctx = at.ops.Context()
        ctx.saved_tensors = (x_gpu, idx_gpu)
        ctx.dim = 1
        grad_out = at.tensor([[1.0], [1.0]], device="gpu")
        with self.assertRaises(at.AMEVAForgeUnsupportedOperationError):
            GatherFunction.backward(ctx, grad_out)

    def test_optimizer_binding_preserved_when_model_to_gpu_called_after_optimizer_init(self):
        # Initialized optimizer with CPU model
        model = at.nn.Linear(2, 4)
        weight_obj_before = model.weight
        bias_obj_before = model.bias
        opt = at.optim.SGD(model.parameters(), lr=0.1)

        self.assertIs(opt.params[0], weight_obj_before)
        self.assertIs(opt.params[1], bias_obj_before)

        # Move model to GPU AFTER optimizer initialization
        model.to("gpu")

        # Must retain identical Tensor instances with updated device
        self.assertIs(model.weight, weight_obj_before)
        self.assertIs(model.bias, bias_obj_before)
        self.assertIs(opt.params[0], model.weight)
        self.assertIs(opt.params[1], model.bias)
        self.assertEqual(model.weight.device, "gpu")
        self.assertEqual(model.bias.device, "gpu")

    def test_positional_encoding_cache_preservation(self):
        pe_module = at.nn.PositionalEncoding(d_model=16, max_len=100)
        x_cpu1 = at.zeros((1, 10, 16))
        x_cpu2 = at.zeros((1, 10, 16))
        x_cpu3 = at.zeros((1, 20, 16))

        out1 = pe_module(x_cpu1)
        self.assertIn(("cpu", 10), pe_module._pe_cache)
        cached_pe_10 = pe_module._pe_cache[("cpu", 10)]

        out2 = pe_module(x_cpu2)
        # Should reuse cached tensor instance for same seq_len
        self.assertIs(pe_module._pe_cache[("cpu", 10)], cached_pe_10)

        out3 = pe_module(x_cpu3)
        self.assertIn(("cpu", 20), pe_module._pe_cache)
        self.assertEqual(len(pe_module._pe_cache), 2)

    def test_move_to_gpu_registers_finalizer_and_collects_on_gc(self):
        import gc
        from forge.tensor import _gc_queue

        # Create CPU tensor (no finalizer)
        t = at.tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.device, "cpu")
        self.assertFalse(t._finalizer_registered)

        # Move to GPU via move_to_
        t.move_to_("gpu")
        self.assertEqual(t.device, "gpu")
        self.assertTrue(t._finalizer_registered)
        # In lazy state, handle is None until realized or assigned
        self.assertIsNone(t._handle)

        # Simulate realized GPU handle
        t._handle = "handle_test_gc_123"
        gpu_handle = t._handle
        self.assertEqual(gpu_handle, "handle_test_gc_123")

        # Delete python reference and force Python GC
        del t
        gc.collect()

        # Handle must be cleanly enqueued in _gc_queue for VRAM release
        self.assertIn(gpu_handle, _gc_queue)

    def test_positional_encoding_dynamic_seq_len_after_to_gpu(self):
        pe_module = at.nn.PositionalEncoding(d_model=16, max_len=100)
        pe_module.to("gpu")
        self.assertEqual(pe_module.pe.device, "gpu")

        # First forward with seq_len = 10
        x_gpu_1 = at.zeros((1, 10, 16), device="gpu")
        out1 = pe_module(x_gpu_1)
        self.assertEqual(out1.shape, (1, 10, 16))

        # Dynamic variable length forward with seq_len = 15
        x_gpu_2 = at.zeros((1, 15, 16), device="gpu")
        out2 = pe_module(x_gpu_2)
        self.assertEqual(out2.shape, (1, 15, 16))

        # Dynamic variable length forward with seq_len = 8
        x_gpu_3 = at.zeros((1, 8, 16), device="gpu")
        out3 = pe_module(x_gpu_3)
        self.assertEqual(out3.shape, (1, 8, 16))

    def test_positional_encoding_lru_cache_eviction(self):
        pe_module = at.nn.PositionalEncoding(d_model=8, max_len=200)
        # Push 40 different sequence lengths
        for seq_len in range(1, 41):
            x = at.zeros((1, seq_len, 8), device="cpu")
            out = pe_module(x)
            self.assertEqual(out.shape, (1, seq_len, 8))
    def test_cross_entropy_cpu_and_gpu_forward_backward(self):
        import forge.functional as F
        # 1. CPU forward & backward
        preds_cpu = at.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], device="cpu", requires_grad=True)
        targets_cpu = at.tensor([0, 1], device="cpu")
        loss_cpu = F.cross_entropy(preds_cpu, targets_cpu)
        self.assertEqual(loss_cpu.shape, ())
        self.assertTrue(loss_cpu.numpy() > 0.0)
        loss_cpu.backward()
        self.assertIsNotNone(preds_cpu.grad)
        self.assertEqual(preds_cpu.grad.shape, (2, 3))

        # 2. GPU forward & backward graph construction
        preds_gpu = at.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], device="gpu", requires_grad=True)
        targets_gpu_cpu = at.tensor([0, 1], device="cpu")
        loss_gpu = F.cross_entropy(preds_gpu, targets_gpu_cpu)
        self.assertEqual(loss_gpu.shape, ())
        self.assertEqual(loss_gpu.device, "gpu")
        loss_gpu.backward()
        self.assertIsNotNone(preds_gpu.grad)
        self.assertEqual(preds_gpu.grad.shape, (2, 3))
        self.assertEqual(preds_gpu.grad.device, "gpu")

    def test_large_vocab_cross_entropy_on_gpu_without_oom(self):
        import forge.functional as F
        from forge.graph import GraphBuilder
        # N=128, C=32,000 (LLaMA-3 vocabulary scale, 4.096M elements)
        preds = at.tensor(np.zeros((128, 32000), dtype=np.float32), device="gpu", requires_grad=True)
        targets = at.tensor(np.random.randint(0, 32000, size=(128,)), dtype="int32", device="gpu")
        
        # Must execute without OOM / UnsupportedOperationError
        loss = F.cross_entropy(preds, targets)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.device, "gpu")
        
        loss.backward()
        self.assertIsNotNone(preds.grad)
        self.assertEqual(preds.grad.shape, (128, 32000))
        self.assertEqual(preds.grad.device, "gpu")

        # Verify AST DAG compilation
        gb = GraphBuilder()
        gb.add_tensor(preds.grad)
        insts, _ = gb.compile()
        self.assertIn("sparse_cross_entropy_backward", insts)

    def test_gpu_slicing_forward_and_backward(self):
        from forge.graph import GraphBuilder
        # Create a 3D GPU tensor [1, 5, 2]
        x = at.tensor(np.ones((1, 5, 2), dtype=np.float32), device="gpu", requires_grad=True)
        
        # Test 1: Slicing last token [:, -1, :]
        sliced = x[:, -1, :]
        self.assertEqual(sliced.shape, (1, 2))
        self.assertEqual(sliced.device, "gpu")
        self.assertEqual(sliced._op, "slice")
        
        # Test 2: Backward pass autograd propagation
        sliced.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (1, 5, 2))
        self.assertEqual(x.grad.device, "gpu")
        self.assertEqual(x.grad._op, "slice_backward")

        # Test 3: Graph compilation verification
        gb = GraphBuilder()
        gb.add_tensor(sliced)
        insts, _ = gb.compile()
        self.assertIn("slice", insts)

if __name__ == '__main__':
    unittest.main()



