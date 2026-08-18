"""
==============================================================================
AMEVA-Forge Neural Models Hub (forge.models)
==============================================================================

WHAT:
  Standard Neural Network Architecture implementations designed for
  zero-copy WebGPU execution, FlashAttention-2, RoPE, RMSNorm, and SwiGLU.

WHY:
  Provides out-of-the-box, production-grade LLM architectures (LLaMA-3, NanoGPT)
  that execute directly in the browser via Pyodide + WebGPU compute shaders.
"""

from .llama import LlamaConfig, LlamaForCausalLM, LlamaModel
from .nanogpt import GPTConfig, GPT, CausalSelfAttention

__all__ = [
    'LlamaConfig',
    'LlamaForCausalLM',
    'LlamaModel',
    'GPTConfig',
    'GPT',
    'CausalSelfAttention',
]
