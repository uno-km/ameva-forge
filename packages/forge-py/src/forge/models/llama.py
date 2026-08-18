"""
==============================================================================
LLaMA-3 Architecture for WebGPU (forge.models.llama)
==============================================================================

WHAT:
  A pure Python, PyTorch-compatible implementation of Meta's LLaMA-3 architecture
  accelerated with WebGPU Native Kernels:
  - FlashAttention-2 1-pass online softmax
  - Rotary Position Embeddings (RoPE)
  - RMS Normalization (RMSNorm)
  - SwiGLU Gated Feed-Forward Networks

WHY:
  Enables running real small-scale language models (SmolLM, NanoLLaMA) directly
  inside web browser tabs with zero server infrastructure.
"""

import math
from typing import Optional, List, Tuple
from dataclasses import dataclass

import forge as torch
import forge.nn as nn
import forge.functional as F
from forge.tensor import Tensor
from forge.ops import tensor, zeros, ones, randn


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 256
    intermediate_size: int = 688
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dtype: str = "float32"
    device: str = "gpu"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = ones((hidden_size,), requires_grad=True)
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swish_gate = F.swiglu(gate, up)
        return self.down_proj(swish_gate)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, offset_pos: int = 0) -> Tensor:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [B, H, L, d]
        q = q.reshape((B, self.num_heads, L, self.head_dim))
        k = k.reshape((B, self.num_key_value_heads, L, self.head_dim))
        v = v.reshape((B, self.num_key_value_heads, L, self.head_dim))

        # Apply Rotary Position Embeddings (RoPE)
        q = F.rope(q, base_freq=self.rope_theta, offset_pos=offset_pos)
        k = F.rope(k, base_freq=self.rope_theta, offset_pos=offset_pos)

        # FlashAttention / Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        # [B, H, L, d] -> [B, L, H * d]
        attn_output = attn_output.reshape((B, L, self.hidden_size))
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states: Tensor, offset_pos: int = 0) -> Tensor:
        # Pre-LN Self-Attention with Residual Connection
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, offset_pos=offset_pos)
        hidden_states = hidden_states + attn_out

        # Pre-LN SwiGLU MLP with Residual Connection
        normed_mlp = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp)
        hidden_states = hidden_states + mlp_out

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

    async def generate(self, prompt_tokens: List[int], max_new_tokens: int = 20) -> List[int]:
        """
        Autoregressive generation loop supporting non-blocking WebGPU execution.
        """
        generated = list(prompt_tokens)
        device = self.config.device

        for _ in range(max_new_tokens):
            inp = tensor([generated], dtype="int32", device=device)
            logits = self.forward(inp)  # [1, L, vocab_size]
            last_token_logits = logits[0, -1, :]
            
            # Read back logits to select top token
            np_logits = await last_token_logits.numpy_async()
            next_token = int(np_logits.argmax())
            generated.append(next_token)

        return generated
