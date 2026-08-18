"""
==============================================================================
NanoGPT Architecture for WebGPU (forge.models.nanogpt)
==============================================================================

WHAT:
  A clean, educational transformer language model implementing Karpathy's NanoGPT,
  optimized with AMEVA-Forge WebGPU compute shaders and in-place autodiff.
"""

import math
from typing import List, Optional
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

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = self.q_proj(x).reshape((B, T, self.n_head, self.head_dim)).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape((B, T, self.n_head, self.head_dim)).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape((B, T, self.n_head, self.head_dim)).permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        y = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.permute(0, 2, 1, 3).reshape((B, T, C))
        return self.c_proj(y)


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

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(tok_emb)
        x = pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
