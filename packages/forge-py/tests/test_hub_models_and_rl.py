"""
==============================================================================
Test Suite: Hub Models (LLaMA-3, NanoGPT), RL (CartPole), Pipeline
==============================================================================
"""

import pytest
import numpy as np
import forge as torch
from forge.models import LlamaConfig, LlamaForCausalLM, GPTConfig, GPT
from forge.rl import CartPoleEnv, DQNAgent
from forge.pipeline import pipeline


def test_nanogpt_forward():
    config = GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32, device="cpu")
    model = GPT(config)
    idx = torch.tensor([[1, 5, 10, 20]], dtype="int32", device="cpu")
    logits = model(idx)
    assert logits.shape == (1, 4, 64)


def test_llama_forward():
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, device="cpu")
    model = LlamaForCausalLM(config)
    idx = torch.tensor([[2, 4, 8]], dtype="int32", device="cpu")
    logits = model(idx)
    assert logits.shape == (1, 3, 64)


def test_cartpole_env():
    env = CartPoleEnv()
    state = env.reset()
    assert state.shape == (4,)
    next_state, reward, done, _ = env.step(1)
    assert next_state.shape == (4,)
    assert isinstance(reward, float)


@pytest.mark.asyncio
async def test_dqn_agent_step():
    env = CartPoleEnv()
    state = env.reset()
    agent = DQNAgent(state_dim=4, hidden_dim=16, action_dim=2, device="cpu")
    action = await agent.act(state, epsilon=0.0)
    assert action in (0, 1)
    next_state, reward, done, _ = env.step(action)
    loss = await agent.train_step(state, action, reward, next_state, done)
    assert isinstance(loss, float)


@pytest.mark.asyncio
async def test_pipeline_text_generation():
    pipe = pipeline("text-generation", device="cpu")
    res = await pipe("Hello", max_new_tokens=3)
    assert "generated_text" in res
    assert len(res["tokens"]) >= 4


@pytest.mark.asyncio
async def test_pipeline_sentiment():
    pipe = pipeline("sentiment-analysis", device="cpu")
    res = await pipe("AMEVA-Forge is blazing fast")
    assert res["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= res["score"] <= 1.0
