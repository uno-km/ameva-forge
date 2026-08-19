"""
AMEVA-Forge Release 2.0 LLM Modules Python Test Suite
Tests RMSNorm, RotaryEmbedding, SwiGLU, and Scaled Dot-Product Attention in forge.nn & forge.functional.
"""

import pytest
import numpy as np
import forge
import forge.nn as nn
import forge.functional as F

class TestRelease2LLMModules:
    def test_rmsnorm_analytical_parity(self):
        eps = 1e-5
        x_np = np.array([[-1.0, 2.0, 3.0, -4.0], [0.5, -0.5, 1.5, -1.5]], dtype=np.float32)
        gamma_np = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
        
        rms_expected = (x_np / np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + eps)) * gamma_np
        
        x = forge.tensor(x_np)
        gamma = forge.tensor(gamma_np)
        out = F.rms_norm(x, gamma, eps=eps)
        
        np.testing.assert_allclose(out.numpy(), rms_expected, atol=1e-5)

    def test_rotary_embedding_analytical_parity(self):
        B, H, N, d = 1, 2, 4, 4
        x_np = np.arange(B * H * N * d, dtype=np.float32).reshape(B, H, N, d) * 0.1
        
        # NumPy RoPE formula
        expected = np.zeros_like(x_np)
        for b in range(B):
            for h in range(H):
                for n in range(N):
                    for k in range(d // 2):
                        theta = (10000.0 ** (-2.0 * k / d)) * n
                        cos_t, sin_t = np.cos(theta), np.sin(theta)
                        v0 = x_np[b, h, n, 2 * k]
                        v1 = x_np[b, h, n, 2 * k + 1]
                        expected[b, h, n, 2 * k] = v0 * cos_t - v1 * sin_t
                        expected[b, h, n, 2 * k + 1] = v1 * cos_t + v0 * sin_t
                        
        x = forge.tensor(x_np)
        out = F.rope(x, base_freq=10000.0, offset_pos=0)
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

    def test_swiglu_analytical_parity(self):
        gate_np = np.array([[0.5, -1.0, 2.0], [-0.5, 1.5, -2.0]], dtype=np.float32)
        up_np = np.array([[1.0, 2.0, -0.5], [0.5, -1.0, 1.5]], dtype=np.float32)
        
        # Swish(gate) * up = (gate * sigmoid(gate)) * up
        sig_gate = 1.0 / (1.0 + np.exp(-gate_np))
        swish_gate = gate_np * sig_gate
        expected_swiglu = swish_gate * up_np
        
        gate = forge.tensor(gate_np)
        up = forge.tensor(up_np)
        out = F.swiglu(gate, up)
        
        np.testing.assert_allclose(out.numpy(), expected_swiglu, atol=1e-5)

    def test_scaled_dot_product_attention_analytical_parity(self):
        B, H, N, d = 1, 2, 4, 8
        scale = 1.0 / np.sqrt(d)
        q_np = np.random.randn(B, H, N, d).astype(np.float32) * 0.5
        k_np = np.random.randn(B, H, N, d).astype(np.float32) * 0.5
        v_np = np.random.randn(B, H, N, d).astype(np.float32) * 0.5
        
        # NumPy Reference SDPA with causal mask
        scores = np.matmul(q_np, k_np.swapaxes(-2, -1)) * scale
        mask = np.triu(np.full((N, N), -1e9, dtype=np.float32), k=1)
        scores += mask
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        expected_out = np.matmul(probs, v_np)
        
        q = forge.tensor(q_np)
        k = forge.tensor(k_np)
        v = forge.tensor(v_np)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        np.testing.assert_allclose(out.numpy(), expected_out, atol=1e-4)

    def test_gpu_rmsnorm_lazy_dag(self):
        x = forge.tensor(np.random.randn(2, 10, 64).astype(np.float32), device='gpu')
        w = forge.tensor(np.ones(64, dtype=np.float32), device='gpu')
        out = F.rms_norm(x, w, eps=1e-5)
        
        assert out.device == 'gpu'
        assert out.shape == (2, 10, 64)
        assert out._lazy_op == 'rmsnorm'
        assert len(out._parents) == 2
        assert out._lazy_params == [1e-5]

    def test_gpu_rope_lazy_dag(self):
        x = forge.tensor(np.random.randn(2, 4, 16, 64).astype(np.float32), device='gpu')
        out = F.rope(x, base_freq=10000.0, offset_pos=4)
        
        assert out.device == 'gpu'
        assert out.shape == (2, 4, 16, 64)
        assert out._lazy_op == 'rope'
        assert len(out._parents) == 1
        assert out._lazy_params == [10000.0, 4.0]

    def test_gpu_swiglu_lazy_dag(self):
        gate = forge.tensor(np.random.randn(4, 128).astype(np.float32), device='gpu')
        up = forge.tensor(np.random.randn(4, 128).astype(np.float32), device='gpu')
        out = F.swiglu(gate, up)
        
        assert out.device == 'gpu'
        assert out.shape == (4, 128)
        assert out._lazy_op == 'swiglu'
        assert len(out._parents) == 2

    def test_gpu_sdpa_lazy_dag(self):
        B, H, N, d = 2, 4, 8, 32
        q = forge.tensor(np.random.randn(B, H, N, d).astype(np.float32), device='gpu')
        k = forge.tensor(np.random.randn(B, H, N, d).astype(np.float32), device='gpu')
        v = forge.tensor(np.random.randn(B, H, N, d).astype(np.float32), device='gpu')
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.device == 'gpu'
        assert out.shape == (B, H, N, d)
        assert out._lazy_op == 'flash_attention'
        assert len(out._parents) == 3
        assert out._lazy_params[2] == 1.0  # is_causal flag
