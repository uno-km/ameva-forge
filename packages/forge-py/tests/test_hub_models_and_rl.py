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
    from forge.tensor import build_lazy_topo
    from forge.graph import GraphBuilder
    import json

    # 1. LLaMA-3 GPU DAG & AST Compilation
    llama_cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, device="gpu")
    llama = LlamaForCausalLM(llama_cfg)
    idx_gpu = torch.tensor([[1, 2, 3]], dtype="int32", device="gpu")
    llama_out = llama(idx_gpu)
    assert llama_out.device == "gpu"
    assert llama_out.shape == (1, 3, 32)
    assert llama_out._lazy_op is not None

    topo_llama = build_lazy_topo(llama_out)
    builder_llama = GraphBuilder()
    node_id_map = {}
    for v in topo_llama:
        if v._handle is not None:
            node_id_map[id(v)] = builder_llama.add_load(v.shape, v._handle)
        elif v._lazy_op == 'upload':
            node_id_map[id(v)] = builder_llama.add_upload(v.shape, v._data)
        else:
            in_ids = [node_id_map[id(p)] for p in v._parents]
            node_id_map[id(v)] = builder_llama.add_op(v._lazy_op, v.shape, in_ids, v._lazy_params)
    insts_json, _ = builder_llama.compile()
    insts = json.loads(insts_json)

    # Verify native embedding node exists with valid inputs and shapes
    embedding_nodes = [n for n in insts if n.get("op") == "embedding"]
    assert len(embedding_nodes) >= 1
    emb_node = embedding_nodes[0]
    assert len(emb_node["in"]) == 2
    assert emb_node["shape"] == [1, 3, 16]
    assert emb_node["params"] == [3, 16, 32, 0]

    # 2. NanoGPT GPU DAG & AST Compilation
    gpt_cfg = GPTConfig(block_size=8, vocab_size=32, n_layer=1, n_head=2, n_embd=16, device="gpu")
    gpt = GPT(gpt_cfg)
    gpt_out = gpt(idx_gpu)
    assert gpt_out.device == "gpu"
    assert gpt_out.shape == (1, 3, 32)
    assert gpt_out._lazy_op is not None

    topo_gpt = build_lazy_topo(gpt_out)
    builder_gpt = GraphBuilder()
    node_id_map_gpt = {}
    for v in topo_gpt:
        if v._handle is not None:
            node_id_map_gpt[id(v)] = builder_gpt.add_load(v.shape, v._handle)
        elif v._lazy_op == 'upload':
            node_id_map_gpt[id(v)] = builder_gpt.add_upload(v.shape, v._data)
        else:
            in_ids = [node_id_map_gpt[id(p)] for p in v._parents]
            node_id_map_gpt[id(v)] = builder_gpt.add_op(v._lazy_op, v.shape, in_ids, v._lazy_params)
    gpt_insts_json, _ = builder_gpt.compile()
    gpt_insts = json.loads(gpt_insts_json)
    gpt_emb_nodes = [n for n in gpt_insts if n.get("op") == "embedding"]
    assert len(gpt_emb_nodes) >= 1

    # 3. PolicyGradientAgent GPU DAG
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
async def test_pipeline_sentiment():
    pipe = pipeline("sentiment-analysis", device="cpu")
    res = await pipe("AMEVA-Forge is blazing fast")
    assert res["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= res["score"] <= 1.0

