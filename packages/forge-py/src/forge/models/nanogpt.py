"""
==============================================================================
NanoGPT Architecture for WebGPU (forge.models.nanogpt)
==============================================================================

WHAT:
  A clean, educational transformer language model implementing Karpathy's NanoGPT,
  optimized with AMEVA-Forge WebGPU compute shaders, KV-Cache linear acceleration,
  and in-place autodiff.
"""

import math
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass

import forge as torch
import forge.nn as nn
import forge.functional as F
from forge.tensor import Tensor
from forge.ops import tensor, zeros, ones


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    device: str = "gpu"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, T, C = x.shape
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = self.q_proj(x).reshape((B, T, self.n_head, self.head_dim)).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape((B, T, self.n_head, self.head_dim)).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape((B, T, self.n_head, self.head_dim)).permute(0, 2, 1, 3)

        if past_key_value is not None:
            from forge.ops import cat
            k = cat([past_key_value[0], k], dim=2)
            v = cat([past_key_value[1], v], dim=2)

        present_kv = (k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        y = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.permute(0, 2, 1, 3).reshape((B, T, C))
        return self.c_proj(y), present_kv


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        attn_out, present_kv = self.attn(self.ln_1(x), past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.PositionalEncoding(config.n_embd, max_len=config.block_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.to(config.device)

    def forward(
        self,
        idx: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False
    ) -> Any:
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(tok_emb)
        x = pos_emb

        present_kvs = []
        for i, block in enumerate(self.blocks):
            p_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_key_value=p_kv)
            present_kvs.append(present_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if use_cache or past_key_values is not None:
            return logits, present_kvs
        return logits

    async def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        use_cache: bool = True
    ) -> List[int]:
        """
        Autoregressive generation loop with O(N) WebGPU KV-Cache linear acceleration.
        """
        import numpy as np
        generated = list(prompt_tokens)
        device = self.config.device

        if not use_cache:
            for _ in range(max_new_tokens):
                inp = tensor([generated[-self.config.block_size:]], dtype="int32", device=device)
                logits = self.forward(inp)
                np_logits = await logits.numpy_async()
                next_token = int(np_logits[0, -1, :].argmax())
                generated.append(next_token)
            return generated

        # --- O(N) Linear Time KV-Cache Accelerated Generation ---
        # 1. Prefill Step
        inp = tensor([prompt_tokens], dtype="int32", device=device)
        logits, past_kvs = self.forward(inp, use_cache=True)
        np_logits = await logits.numpy_async()
        next_token = int(np_logits[0, -1, :].argmax())
        generated.append(next_token)

        # 2. Decode Steps
        for _ in range(1, max_new_tokens):
            inp = tensor([[next_token]], dtype="int32", device=device)
            logits, past_kvs = self.forward(inp, past_key_values=past_kvs, use_cache=True)
            np_logits = await logits.numpy_async()
            next_token = int(np_logits[0, -1, :].argmax())
            generated.append(next_token)

        return generated
