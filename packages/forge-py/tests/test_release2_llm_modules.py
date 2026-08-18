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
    def test_rmsnorm_module(self):
        norm = nn.RMSNorm(64)
        x = forge.tensor(np.random.randn(2, 10, 64).astype(np.float32))
        out = norm(x)
        
        assert out.shape == (2, 10, 64)
        out_np = out.numpy()
        
        # Verify RMS of output is approximately 1.0 along last axis
        mean_sq = np.mean(out_np ** 2, axis=-1)
        np.testing.assert_allclose(np.sqrt(mean_sq), 1.0, atol=1e-2)

    def test_rotary_embedding_module(self):
        rope_layer = nn.RotaryEmbedding(64)
        x = forge.tensor(np.random.randn(2, 4, 16, 64).astype(np.float32))
        out = rope_layer(x)
        
        assert out.shape == (2, 4, 16, 64)
        # Position 0 should not change magnitude
        np.testing.assert_allclose(np.linalg.norm(out.numpy(), axis=-1), np.linalg.norm(x.numpy(), axis=-1), rtol=1e-4)

    def test_swiglu_module(self):
        swiglu_block = nn.SwiGLU(128, 256)
        x = forge.tensor(np.random.randn(4, 128).astype(np.float32))
        out = swiglu_block(x)
        
        assert out.shape == (4, 128)
        assert np.all(np.isfinite(out.numpy()))

    def test_scaled_dot_product_attention_functional(self):
        B, H, N, d = 2, 4, 8, 32
        q = forge.tensor(np.random.randn(B, H, N, d).astype(np.float32))
        k = forge.tensor(np.random.randn(B, H, N, d).astype(np.float32))
        v = forge.tensor(np.random.randn(B, H, N, d).astype(np.float32))
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (B, H, N, d)
        assert np.all(np.isfinite(out.numpy()))

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
