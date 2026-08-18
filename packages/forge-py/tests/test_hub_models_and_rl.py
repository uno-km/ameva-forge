"""
==============================================================================
Test Suite: Hub Models (LLaMA-3, NanoGPT), RL (CartPole), Pipeline
==============================================================================
"""

import pytest
import numpy as np
import forge as torch
from forge.models import LlamaConfig, LlamaForCausalLM, GPTConfig, GPT
from forge.rl import CartPoleEnv, DQNAgent, PolicyGradientAgent
from forge.pipeline import pipeline
from forge.graph import GraphBuilder
from forge.bridge import is_webgpu_available


def test_nanogpt_forward_cpu_and_numerical_invariants():
    config = GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32, device="cpu")
    model = GPT(config)
    idx = torch.tensor([[1, 5, 10, 20]], dtype="int32", device="cpu")
    logits = model(idx)
    assert logits.shape == (1, 4, 64)
    assert logits.device == "cpu"
    
    # Mathematical Invariant: Softmax distribution must sum to 1.0 per token without NaN/Inf
    logits_np = logits.numpy()
    assert not np.isnan(logits_np).any()
    assert not np.isinf(logits_np).any()
    
    probs = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones((1, 4)), atol=1e-5)


def test_llama_forward_cpu_and_causal_invariance():
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, device="cpu")
    model = LlamaForCausalLM(config)
    idx1 = torch.tensor([[2, 4]], dtype="int32", device="cpu")
    idx2 = torch.tensor([[2, 4, 8, 16]], dtype="int32", device="cpu")
    
    logits1 = model(idx1)
    logits2 = model(idx2)
    assert logits1.shape == (1, 2, 64)
    assert logits2.shape == (1, 4, 64)

    # Invariant: Logits at token 0 and 1 must be identical regardless of subsequent tokens (Causal Masking Proof)
    np.testing.assert_allclose(logits1.numpy()[:, :2, :], logits2.numpy()[:, :2, :], atol=1e-4)


def test_policy_gradient_agent_forward_and_action():
    agent = PolicyGradientAgent(state_dim=4, hidden_dim=16, action_dim=2, device="cpu")
    state = torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype="float32", device="cpu")
    probs = agent(state)
    assert probs.shape == (1, 2)
    assert probs.device == "cpu"
    probs_np = probs.numpy()
    assert np.all(probs_np >= 0.0)
    assert np.all(probs_np <= 1.0)
    np.testing.assert_allclose(np.sum(probs_np, axis=-1), [1.0], atol=1e-5)


def test_models_on_gpu_lazy_dag():
    # 1. NanoGPT GPU DAG
    gpt_cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2, n_embd=32, device="gpu")
    gpt_model = GPT(gpt_cfg)
    idx_gpu = torch.tensor([[1, 2, 3]], dtype="int32", device="gpu")
    logits_gpu = gpt_model(idx_gpu)
    assert logits_gpu.device == "gpu"
    assert logits_gpu.shape == (1, 3, 64)
    
    # 2. LLaMA GPU DAG (Small sequence)
    llama_cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, device="gpu")
    llama_model = LlamaForCausalLM(llama_cfg)
    llama_out = llama_model(idx_gpu)
    assert llama_out.device == "gpu"
    assert llama_out.shape == (1, 3, 64)

    # Verify GraphBuilder compiles LLaMA WebGPU AST with valid embedding schema
    gb = GraphBuilder()
    gb.add_tensor(llama_out)
    ast = gb.to_dict()
    assert len(ast["instructions"]) >= 1
    # Check that embedding instruction exists with [num_tokens, embedding_dim, vocab_size, 0] params
    embedding_inst = [i for i in ast["instructions"] if i.get("op") == "embedding"]
    assert len(embedding_inst) == 1
    assert embedding_inst[0]["params"] == [3, 32, 64, 0]

    # 3. LLaMA Large Context (N=128 tokens) GPU DAG Invariance
    idx_large_gpu = torch.tensor(np.random.randint(0, 64, size=(1, 128)), dtype="int32", device="gpu")
    llama_large_out = llama_model(idx_large_gpu)
    assert llama_large_out.device == "gpu"
    assert llama_large_out.shape == (1, 128, 64)

    gb_large = GraphBuilder()
    gb_large.add_tensor(llama_large_out)
    ast_large = gb_large.to_dict()
    large_emb = [i for i in ast_large["instructions"] if i.get("op") == "embedding"]
    assert len(large_emb) == 1
    assert large_emb[0]["params"] == [128, 32, 64, 0]

    # 4. PolicyGradientAgent GPU DAG
    pg_agent = PolicyGradientAgent(state_dim=4, hidden_dim=8, action_dim=2, device="gpu")
    st_gpu = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype="float32", device="gpu")
    pg_out = pg_agent(st_gpu)
    assert pg_out.device == "gpu"
    assert pg_out.shape == (1, 2)



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
async def test_pipeline_text_generation_gpu():
    if not is_webgpu_available():
        pytest.skip("WebGPU bridge not available in CPython testing environment")
    pipe = pipeline("text-generation", device="gpu")
    res = await pipe("Hello WebGPU", max_new_tokens=3)
    assert "generated_text" in res
    assert len(res["tokens"]) >= 4


@pytest.mark.asyncio
async def test_pipeline_sentiment():
    pipe = pipeline("sentiment-analysis", device="cpu")
    res = await pipe("AMEVA-Forge is blazing fast")
    assert res["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= res["score"] <= 1.0


@pytest.mark.asyncio
async def test_pipeline_sentiment_gpu():
    if not is_webgpu_available():
        pytest.skip("WebGPU bridge not available in CPython testing environment")
    pipe = pipeline("sentiment-analysis", device="gpu")
    res = await pipe("WebGPU AI inference is state of the art")
    assert res["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= res["score"] <= 1.0


def test_gpu_embedding_autograd_backward():
    # Verify that Embedding on GPU supports backward() producing embedding_backward DAG node
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32", device="gpu", requires_grad=True)
    idx = torch.tensor([0, 1], dtype="int32", device="gpu")
    out = torch.ops.embedding(weight, idx)
    assert out.device == "gpu"
    assert out.shape == (2, 2)

    # Backward pass
    grad_out = torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype="float32", device="gpu")
    out.backward(grad_out)

    assert weight.grad is not None
    assert weight.grad.device == "gpu"
    assert weight.grad.shape == (2, 2)
    assert weight.grad._lazy_op == "embedding_backward"
    assert weight.grad._lazy_params == [2, 2, 2, 4]

    # Verify GraphBuilder compiles embedding_backward AST without errors
    gb = GraphBuilder()
    gb.add_tensor(weight.grad)
    ast = gb.to_dict()
    assert len(ast["instructions"]) >= 1
    backward_inst = [i for i in ast["instructions"] if i.get("op") == "embedding_backward"]
    assert len(backward_inst) == 1
    assert backward_inst[0]["params"] == [2, 2, 2, 4]
