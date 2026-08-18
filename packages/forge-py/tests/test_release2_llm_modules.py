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
