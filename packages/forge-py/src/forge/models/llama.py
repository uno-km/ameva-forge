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
from typing import Optional, List, Tuple, Any, Union
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

    def forward(
        self,
        hidden_states: Tensor,
        offset_pos: int = 0,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [B, L, H * d] -> [B, L, H, d] -> [B, H, L, d] (Permute to separate heads properly)
        q = q.reshape((B, L, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)
        k = k.reshape((B, L, self.num_key_value_heads, self.head_dim)).permute(0, 2, 1, 3)
        v = v.reshape((B, L, self.num_key_value_heads, self.head_dim)).permute(0, 2, 1, 3)

        # Apply Rotary Position Embeddings (RoPE)
        q = F.rope(q, base_freq=self.rope_theta, offset_pos=offset_pos)
        k = F.rope(k, base_freq=self.rope_theta, offset_pos=offset_pos)

        # Append to KV-Cache if provided
        if past_key_value is not None:
            from forge.ops import cat
            k = cat([past_key_value[0], k], dim=2)
            v = cat([past_key_value[1], v], dim=2)

        present_key_value = (k, v)

        # FlashAttention / Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        # [B, H, L, d] -> [B, L, H, d] -> [B, L, H * d] (Restore sequence-contiguous layout)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape((B, L, self.hidden_size))
        return self.o_proj(attn_output), present_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        offset_pos: int = 0,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # Pre-LN Self-Attention with Residual Connection
        normed = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(normed, offset_pos=offset_pos, past_key_value=past_key_value)
        hidden_states = hidden_states + attn_out

        # Pre-LN SwiGLU MLP with Residual Connection
        normed_mlp = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_mlp)
        hidden_states = hidden_states + mlp_out

        return hidden_states, present_kv


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        offset_pos: int = 0
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        present_kvs = []
        for idx, layer in enumerate(self.layers):
            p_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = layer(hidden_states, offset_pos=offset_pos, past_key_value=p_kv)
            present_kvs.append(present_kv)
        return self.norm(hidden_states), present_kvs


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.to(config.device)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        offset_pos: int = 0,
        use_cache: bool = False
    ) -> Any:
        hidden_states, present_kvs = self.model(input_ids, past_key_values=past_key_values, offset_pos=offset_pos)
        logits = self.lm_head(hidden_states)
        if use_cache or past_key_values is not None:
            return logits, present_kvs
        return logits

    async def generate(self, prompt_tokens: List[int], max_new_tokens: int = 20, use_cache: bool = True) -> List[int]:
        """
        Autoregressive generation loop with O(N) WebGPU KV-Cache acceleration.
        """
        generated = list(prompt_tokens)
        device = self.config.device

        if not use_cache:
            for _ in range(max_new_tokens):
                inp = tensor([generated], dtype="int32", device=device)
                logits = self.forward(inp)  # [1, L, vocab_size]
                last_token_logits = logits[0, -1, :]
                np_logits = await last_token_logits.numpy_async()
                next_token = int(np_logits.argmax())
                generated.append(next_token)
            return generated

        # --- O(N) Linear Time KV-Cache Accelerated Generation ---
        # 1. Prefill Step (Process initial prompt)
        inp = tensor([prompt_tokens], dtype="int32", device=device)
        logits, past_kvs = self.forward(inp, use_cache=True, offset_pos=0)
        last_token_logits = logits[0, -1, :]
        np_logits = await last_token_logits.numpy_async()
        next_token = int(np_logits.argmax())
        generated.append(next_token)

        # 2. Decode Steps (Single token forward per step)
        for step in range(1, max_new_tokens):
            inp = tensor([[next_token]], dtype="int32", device=device)
            offset = len(prompt_tokens) + step - 1
            logits, past_kvs = self.forward(inp, past_key_values=past_kvs, offset_pos=offset, use_cache=True)
            last_token_logits = logits[0, -1, :]
            np_logits = await last_token_logits.numpy_async()
            next_token = int(np_logits.argmax())
            generated.append(next_token)

        return generated

