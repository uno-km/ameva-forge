import pytest
import numpy as np
import forge as fg
from forge.ops import tensor, bmm, matmul, relu, add, sub, mul, div, clone
from forge.functional import softmax, log_softmax, layer_norm, cross_entropy
from forge.errors import AMEVAForgeValidationError, AMEVAForgeShapeError

def test_fuzz_bmm_broadcasting_matrix():
    """Fuzz test various 3D BMM broadcasting combinations."""
    shapes = [
        ((1, 16, 8), (4, 8, 12), (4, 16, 12)),
        ((4, 16, 8), (1, 8, 12), (4, 16, 12)),
        ((2, 5, 7), (2, 7, 9), (2, 5, 9)),
    ]
    for shape_a, shape_b, expected_out in shapes:
        a = tensor(np.random.randn(*shape_a).astype(np.float32), requires_grad=True)
        b = tensor(np.random.randn(*shape_b).astype(np.float32), requires_grad=True)
        c = bmm(a, b)
        assert c.shape == expected_out
        loss = c.sum()
        loss.backward()
        assert a.grad.shape == shape_a
        assert b.grad.shape == shape_b

def test_fuzz_layernorm_arbitrary_normalized_shapes():
    """Fuzz test LayerNorm on various tensor ranks and normalized_shapes."""
    test_configs = [
        ((2, 3, 4), (4,)),
        ((2, 3, 4), (3, 4)),
        ((2, 3, 4, 5), (4, 5)),
        ((2, 3, 4, 5), (3, 4, 5)),
    ]
    for x_shape, norm_shape in test_configs:
        x = tensor(np.random.randn(*x_shape).astype(np.float32), requires_grad=True)
        y = layer_norm(x, norm_shape)
        assert y.shape == x_shape
        y_np = y.numpy()
        dims = tuple(range(-len(norm_shape), 0))
        mean = np.mean(y_np, axis=dims)
        std = np.std(y_np, axis=dims)
        np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-4)
        np.testing.assert_allclose(std, np.ones_like(std), atol=1e-3)

def test_fuzz_all_masked_softmax_stress():
    """Fuzz test Softmax and LogSoftmax with partial and total masked rows."""
    data = np.random.randn(8, 16).astype(np.float32)
    data[0, :] = -np.inf
    data[3, :] = -np.inf
    data[1, :8] = -np.inf
    
    t = tensor(data)
    sm = softmax(t, axis=-1)
    sm_np = sm.numpy()
    assert not np.isnan(sm_np).any()
    assert np.all(sm_np[0] == 0.0)
    assert np.all(sm_np[3] == 0.0)
    np.testing.assert_allclose(np.sum(sm_np[1]), 1.0, atol=1e-5)

    lsm = log_softmax(t, axis=-1)
    lsm_np = lsm.numpy()
    assert not np.isnan(lsm_np).any()

def test_fuzz_non_differentiable_dtypes():
    invalid_dtypes = ["int8", "int16", "int32", "int64", "uint8", "bool"]
    for dt in invalid_dtypes:
        with pytest.raises(AMEVAForgeValidationError):
            tensor([1, 2, 3], dtype=dt, requires_grad=True)

def test_fuzz_cross_entropy_ignore_index_and_3d():
    """Fuzz test CrossEntropy with ignore_index=-100 and 3D LLM shapes."""
    # 1. 2D with ignore_index=-100
    preds = tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
    targets = tensor(np.array([2, -100, 5, -100], dtype=np.int32))
    
    loss = cross_entropy(preds, targets)
    assert not np.isnan(loss.numpy())
    loss.backward()
    assert preds.grad is not None
    assert not np.isnan(preds.grad.numpy()).any()
    # Gradient for rows with target=-100 should be 0.0
    grad_np = preds.grad.numpy()
    np.testing.assert_allclose(grad_np[1], np.zeros_like(grad_np[1]), atol=1e-6)
    np.testing.assert_allclose(grad_np[3], np.zeros_like(grad_np[3]), atol=1e-6)

    # 2. 3D LLM [Batch=2, Seq=3, Vocab=8]
    preds_3d = tensor(np.random.randn(2, 3, 8).astype(np.float32), requires_grad=True)
    targets_2d = tensor(np.array([[1, 2, -100], [0, -100, 7]], dtype=np.int32))
    loss_3d = cross_entropy(preds_3d, targets_2d)
    assert not np.isnan(loss_3d.numpy())
    loss_3d.backward()
    assert preds_3d.grad is not None
    assert not np.isnan(preds_3d.grad.numpy()).any()

def test_fuzz_embedding_padding_idx_and_bounds():
    """Fuzz test nn.Embedding padding_idx zeroing and out-of-bounds index rejection."""
    import forge.nn as nn
    emb = nn.Embedding(10, 16, padding_idx=0)
    # Weight at padding_idx should be strictly 0
    w_np = emb.weight.numpy()
    np.testing.assert_allclose(w_np[0], np.zeros(16))
    
    # Forward lookup
    idx = tensor(np.array([0, 3, 5, 0], dtype=np.int32))
    out = emb(idx)
    assert out.shape == (4, 16)
    np.testing.assert_allclose(out.numpy()[0], np.zeros(16))
    np.testing.assert_allclose(out.numpy()[3], np.zeros(16))
    
    # Backward gradient at padding_idx must stay 0
    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None
    grad_w = emb.weight.grad.numpy()
    np.testing.assert_allclose(grad_w[0], np.zeros(16))
    
    # Out of bounds index should raise IndexError
    with pytest.raises(IndexError):
        emb(tensor(np.array([12], dtype=np.int32)))
    with pytest.raises(IndexError):
        emb(tensor(np.array([-1], dtype=np.int32)))

def test_fuzz_rms_norm_multidim_and_signatures():
    """Fuzz test RMSNorm multi-dimensional normalized_shape and dual API signatures."""
    import forge.functional as F
    import forge.nn as nn
    
    # 1. Multi-dimensional reduction (B=2, H=3, W=4, norm_shape=(3, 4))
    x = tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    norm = nn.RMSNorm((3, 4), eps=1e-5)
    y = norm(x)
    assert y.shape == (2, 3, 4)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not np.isnan(x.grad.numpy()).any()

    # 2. PyTorch 2.4 signature F.rms_norm(x, normalized_shape, weight, eps)
    w = tensor(np.ones((4,)).astype(np.float32), requires_grad=True)
    x1 = tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    y1 = F.rms_norm(x1, (4,), weight=w, eps=1e-5)
    assert y1.shape == (2, 3, 4)

    # 3. LLaMA signature F.rms_norm(x, weight, eps)
    x2 = tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    y2 = F.rms_norm(x2, w, 1e-5)
    assert y2.shape == (2, 3, 4)

def test_fuzz_multihead_attention_masks_and_invariants():
    """Fuzz test MultiheadAttention divisibility validation and key_padding_mask."""
    import forge.nn as nn
    
    # 1. Divisibility invariant failure
    with pytest.raises(AMEVAForgeValidationError):
        nn.MultiheadAttention(embed_dim=65, num_heads=8)
        
    # 2. Key padding mask execution
    mha = nn.MultiheadAttention(embed_dim=32, num_heads=4)
    q = tensor(np.random.randn(2, 6, 32).astype(np.float32), requires_grad=True)
    k = tensor(np.random.randn(2, 6, 32).astype(np.float32), requires_grad=True)
    v = tensor(np.random.randn(2, 6, 32).astype(np.float32), requires_grad=True)
    
    # Mask last 2 tokens of batch 0
    key_padding_mask = tensor(np.array([[False, False, False, False, True, True],
                                        [False, False, False, False, False, False]]))
    out = mha(q, k, v, key_padding_mask=key_padding_mask)
    assert out.shape == (2, 6, 32)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert not np.isnan(q.grad.numpy()).any()

def test_fuzz_llama_gqa_and_mismatched_kv_heads():
    """Fuzz test LLaMA Grouped Query Attention (GQA) with 4x query heads vs KV heads."""
    from forge.models.llama import LlamaConfig, LlamaAttention
    
    # 8 Query heads, 2 KV heads (GQA with ratio 4:1)
    cfg = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        device="cpu"
    )
    attn = LlamaAttention(cfg)
    x = tensor(np.random.randn(2, 5, 64).astype(np.float32), requires_grad=True)
    out, past_kv = attn(x)
    assert out.shape == (2, 5, 64)
    assert past_kv[0].shape == (2, 2, 5, 8) # (B, num_kv_heads, S, head_dim)
    
    # Backward pass verification
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not np.isnan(x.grad.numpy()).any()

def test_fuzz_batch_norm2d_affine_false_and_no_running_stats():
    """Fuzz test BatchNorm2d with affine=False, track_running_stats=False, and shape invariants."""
    import forge.nn as nn
    from forge.errors import AMEVAForgeShapeError
    
    # 1. 3D input shape error
    bn = nn.BatchNorm2d(num_features=16)
    with pytest.raises(AMEVAForgeShapeError):
        bn(tensor(np.random.randn(2, 16, 4).astype(np.float32)))
        
    # 2. affine=False, track_running_stats=False
    bn_stateless = nn.BatchNorm2d(num_features=8, affine=False, track_running_stats=False)
    x = tensor(np.random.randn(4, 8, 7, 7).astype(np.float32), requires_grad=True)
    out = bn_stateless(x)
    assert out.shape == (4, 8, 7, 7)
    assert bn_stateless.weight is None
    assert bn_stateless.bias is None
    assert bn_stateless.running_mean is None
    assert bn_stateless.running_var is None
    
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not np.isnan(x.grad.numpy()).any()

def test_fuzz_cosine_similarity_clamp_and_maximum():
    """Fuzz test cosine similarity, clamp, maximum/minimum autograd."""
    import forge.functional as F
    import forge.nn as nn
    
    # 1. Cosine similarity zero-division defense
    x1 = tensor(np.zeros((3, 5), dtype=np.float32), requires_grad=True)
    x2 = tensor(np.zeros((3, 5), dtype=np.float32), requires_grad=True)
    sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
    assert sim.shape == (3,)
    assert not np.isnan(sim.numpy()).any()
    loss = sim.sum()
    loss.backward()
    assert x1.grad is not None
    assert not np.isnan(x1.grad.numpy()).any()
    
    # 2. nn.CosineSimilarity module
    cos_mod = nn.CosineSimilarity(dim=1)
    a = tensor(np.random.randn(4, 16).astype(np.float32), requires_grad=True)
    b = tensor(np.random.randn(4, 16).astype(np.float32), requires_grad=True)
    res = cos_mod(a, b)
    assert res.shape == (4,)
    
    # 3. Clamp / Maximum / Minimum
    v = tensor(np.array([-5.0, 0.0, 5.0, 10.0], dtype=np.float32), requires_grad=True)
    clamped = v.clamp(min_val=0.0, max_val=6.0)
    np.testing.assert_allclose(clamped.numpy(), np.array([0.0, 0.0, 5.0, 6.0], dtype=np.float32))
    loss_c = clamped.sum()
    loss_c.backward()
    assert v.grad is not None
    np.testing.assert_allclose(v.grad.numpy(), np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))

def test_fuzz_clip_grad_norm_generator_and_p_norms():
    """Fuzz test clip_grad_norm_ with generator inputs, Inf-norm, and nn.utils namespace."""
    import forge.nn as nn
    
    # 1. Generator parameter input (e.g. (p for p in ...))
    p1 = tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
    p1.grad = tensor(np.array([3.0, 4.0], dtype=np.float32)) # norm = 5.0
    
    # Pass as generator
    gen = (p for p in [p1])
    total_norm = nn.utils.clip_grad_norm_(gen, max_norm=1.0)
    assert total_norm == 5.0
    # Gradient must be scaled to [3/5, 4/5] = [0.6, 0.8]
    np.testing.assert_allclose(p1.grad.numpy(), np.array([0.6, 0.8], dtype=np.float32), atol=1e-5)
    
    # 2. Inf norm
    p2 = tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p2.grad = tensor(np.array([10.0], dtype=np.float32))
    inf_norm = nn.utils.clip_grad_norm_([p2], max_norm=2.0, norm_type=float('inf'))
    assert inf_norm == 10.0
    np.testing.assert_allclose(p2.grad.numpy(), np.array([2.0], dtype=np.float32), atol=1e-5)

def test_fuzz_tensor_creation_and_triangular_ops():
    """Fuzz test arange, eye, linspace, triu, tril autograd & masks."""
    import forge as at
    
    # 1. arange
    r1 = at.arange(5)
    np.testing.assert_allclose(r1.numpy(), np.array([0, 1, 2, 3, 4], dtype=np.float32))
    r2 = at.arange(2, 10, 2)
    np.testing.assert_allclose(r2.numpy(), np.array([2, 4, 6, 8], dtype=np.float32))
    
    # 2. eye
    i_mat = at.eye(3)
    np.testing.assert_allclose(i_mat.numpy(), np.eye(3, dtype=np.float32))
    
    # 3. linspace
    ls = at.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(ls.numpy(), np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32))
    
    # 4. triu & tril autograd
    mat = tensor(np.ones((3, 3), dtype=np.float32), requires_grad=True)
    u = mat.triu(diagonal=1)
    np.testing.assert_allclose(u.numpy(), np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float32))
    loss_u = u.sum()
    loss_u.backward()
    np.testing.assert_allclose(mat.grad.numpy(), np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float32))









