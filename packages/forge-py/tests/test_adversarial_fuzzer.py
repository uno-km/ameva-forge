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

def test_fuzz_activations_gelu_silu_leaky_elu_autograd():
    """Fuzz test GELU (none/tanh), SiLU, LeakyReLU, ELU modules & autograd."""
    import forge as at
    import forge.nn as nn
    import forge.functional as F
    
    # 1. GELU exact and tanh approximation
    x_g = at.tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32), requires_grad=True)
    g_out = F.gelu(x_g, approximate="tanh")
    assert not np.isnan(g_out.numpy()).any()
    loss_g = g_out.sum()
    loss_g.backward()
    assert x_g.grad is not None
    assert not np.isnan(x_g.grad.numpy()).any()
    
    # 2. SiLU module
    silu_mod = nn.SiLU()
    x_s = at.tensor(np.array([-1.0, 0.0, 1.0], dtype=np.float32), requires_grad=True)
    s_out = silu_mod(x_s)
    loss_s = s_out.sum()
    loss_s.backward()
    assert x_s.grad is not None
    assert not np.isnan(x_s.grad.numpy()).any()
    
    # 3. LeakyReLU module
    lrelu = nn.LeakyReLU(negative_slope=0.1)
    x_l = at.tensor(np.array([-5.0, 5.0], dtype=np.float32), requires_grad=True)
    l_out = lrelu(x_l)
    np.testing.assert_allclose(l_out.numpy(), np.array([-0.5, 5.0], dtype=np.float32))
    loss_l = l_out.sum()
    loss_l.backward()
    np.testing.assert_allclose(x_l.grad.numpy(), np.array([0.1, 1.0], dtype=np.float32))
    
    # 4. ELU module
    elu_mod = nn.ELU(alpha=1.0)
    x_e = at.tensor(np.array([0.0, 2.0], dtype=np.float32), requires_grad=True)
    e_out = elu_mod(x_e)
    loss_e = e_out.sum()
    loss_e.backward()
    assert x_e.grad is not None

def test_fuzz_identity_and_pad_modules():
    """Fuzz test nn.Identity and 2D padding modules with autograd."""
    import forge as at
    import forge.nn as nn
    
    # 1. Identity module
    ident = nn.Identity(54, unused_kwarg=True)
    x = at.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    y = ident(x)
    assert y is x
    loss = y.sum()
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), np.ones((2, 2), dtype=np.float32))
    
    # 2. ZeroPad2d / ConstantPad2d / ReflectionPad2d
    zpad = nn.ZeroPad2d((1, 1, 2, 2))
    x_img = at.tensor(np.ones((1, 1, 4, 4), dtype=np.float32), requires_grad=True)
    padded = zpad(x_img)
    assert padded.shape == (1, 1, 8, 6)
    loss_p = padded.sum()
    loss_p.backward()
    np.testing.assert_allclose(x_img.grad.numpy(), np.ones((1, 1, 4, 4), dtype=np.float32))

def test_fuzz_adaptive_pooling_modules_and_autograd():
    """Fuzz test AdaptiveAvgPool2d and AdaptiveMaxPool2d with arbitrary output sizes."""
    import forge as at
    import forge.nn as nn
    
    # 1. Global Adaptive Average Pooling ((1, 1))
    gap = nn.AdaptiveAvgPool2d((1, 1))
    x = at.tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4), requires_grad=True)
    out_gap = gap(x)
    assert out_gap.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(out_gap.numpy(), np.array([[[[7.5]]]], dtype=np.float32))
    loss_gap = out_gap.sum()
    loss_gap.backward()
    np.testing.assert_allclose(x.grad.numpy(), np.full((1, 1, 4, 4), 1.0 / 16.0, dtype=np.float32))
    
    # 2. Adaptive Max Pooling ((2, 2))
    gmp = nn.AdaptiveMaxPool2d((2, 2))
    x2 = at.tensor(np.array([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]]]], dtype=np.float32), requires_grad=True)
    out_gmp = gmp(x2)
    assert out_gmp.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out_gmp.numpy(), np.array([[[[6, 8], [14, 16]]]], dtype=np.float32))
    loss_gmp = out_gmp.sum()
    loss_gmp.backward()
    assert x2.grad is not None
    assert np.sum(x2.grad.numpy()) == 4.0

def test_fuzz_advanced_loss_functions_and_autograd():
    """Fuzz test BCEWithLogitsLoss, SmoothL1Loss, KLDivLoss, L1Loss with Autograd."""
    import forge as at
    import forge.nn as nn
    
    # 1. BCEWithLogitsLoss
    bce = nn.BCEWithLogitsLoss()
    logits = at.tensor(np.array([0.0, 10.0, -10.0], dtype=np.float32), requires_grad=True)
    target = at.tensor(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    loss_bce = bce(logits, target)
    assert not np.isnan(loss_bce.numpy())
    loss_bce.backward()
    assert logits.grad is not None
    assert not np.isnan(logits.grad.numpy()).any()
    
    # 2. SmoothL1Loss (Huber)
    smooth = nn.SmoothL1Loss(beta=1.0)
    pred_s = at.tensor(np.array([0.5, 3.0], dtype=np.float32), requires_grad=True)
    target_s = at.tensor(np.array([0.0, 0.0], dtype=np.float32))
    loss_s = smooth(pred_s, target_s)
    loss_s.backward()
    # 0.5 < 1.0 -> grad is 0.5 / 2 = 0.25; 3.0 >= 1.0 -> grad is 1.0 / 2 = 0.5
    np.testing.assert_allclose(pred_s.grad.numpy(), np.array([0.25, 0.5], dtype=np.float32))
    
    # 3. L1Loss
    l1 = nn.L1Loss()
    pred_l1 = at.tensor(np.array([2.0, -2.0], dtype=np.float32), requires_grad=True)
    target_l1 = at.tensor(np.array([0.0, 0.0], dtype=np.float32))
    loss_l1 = l1(pred_l1, target_l1)
    loss_l1.backward()
    np.testing.assert_allclose(pred_l1.grad.numpy(), np.array([0.5, -0.5], dtype=np.float32))
    
    # 4. KLDivLoss
    kld = nn.KLDivLoss(reduction='batchmean')
    log_probs = at.tensor(np.log(np.array([[0.5, 0.5]], dtype=np.float32)), requires_grad=True)
    targets_prob = at.tensor(np.array([[0.8, 0.2]], dtype=np.float32))
    loss_kld = kld(log_probs, targets_prob)
    loss_kld.backward()
    assert log_probs.grad is not None
    np.testing.assert_allclose(log_probs.grad.numpy(), np.array([[-0.8, -0.2]], dtype=np.float32))

def test_fuzz_topk_sort_argsort_and_autograd():
    """Fuzz test topk, sort, argsort and autograd gradient routing."""
    import forge as at
    
    # 1. topk largest
    x = at.tensor(np.array([[10.0, 50.0, 20.0, 40.0], [5.0, 1.0, 9.0, 3.0]], dtype=np.float32), requires_grad=True)
    res = at.topk(x, k=2, dim=-1, largest=True)
    np.testing.assert_allclose(res.values.numpy(), np.array([[50.0, 40.0], [9.0, 5.0]], dtype=np.float32))
    np.testing.assert_allclose(res.indices.numpy(), np.array([[1, 3], [2, 0]], dtype=np.int32))
    
    loss_topk = res.values.sum()
    loss_topk.backward()
    expected_grad = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected_grad)
    
    # 2. sort and argsort
    y = at.tensor(np.array([30.0, 10.0, 20.0], dtype=np.float32), requires_grad=True)
    s_res = y.sort(descending=False)
    np.testing.assert_allclose(s_res.values.numpy(), np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(s_res.indices.numpy(), np.array([1, 2, 0], dtype=np.int32))
    
    idx = y.argsort(descending=True)
    np.testing.assert_allclose(idx.numpy(), np.array([0, 2, 1], dtype=np.int32))
    
    loss_s = s_res.values.sum()
    loss_s.backward()
    np.testing.assert_allclose(y.grad.numpy(), np.ones(3, dtype=np.float32))

def test_fuzz_predicate_inspection_isnan_isclose_allclose():
    """Fuzz test isnan, isinf, isfinite, isclose, allclose, any, all."""
    import forge as at
    
    # 1. isnan, isinf, isfinite
    t = at.tensor(np.array([1.0, float('nan'), float('inf'), float('-inf'), 0.0], dtype=np.float32))
    assert t.isnan().numpy().tolist() == [False, True, False, False, False]
    assert t.isinf().numpy().tolist() == [False, False, True, True, False]
    assert t.isfinite().numpy().tolist() == [True, False, False, False, True]
    
    # 2. isclose & allclose
    a = at.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = at.tensor(np.array([1.000001, 2.0, 3.00001], dtype=np.float32))
    assert at.allclose(a, b, atol=1e-4)
    assert not at.allclose(a, b, rtol=0.0, atol=1e-7)
    assert a.isclose(b, atol=1e-4).all().numpy()
    
    # 3. any & all
    c = at.tensor(np.array([[True, False], [True, True]]))
    assert c.any().numpy() == True
    assert c.all(dim=1).numpy().tolist() == [False, True]

def test_fuzz_conv1d_and_pixel_transformation_modules():
    """Fuzz test Conv1d, PixelShuffle, PixelUnshuffle, Upsample with Autograd."""
    import forge as at
    import forge.nn as nn
    
    # 1. Conv1d forward & backward
    conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
    x = at.tensor(np.ones((2, 2, 8), dtype=np.float32), requires_grad=True)
    out = conv1(x)
    assert out.shape == (2, 4, 8)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert conv1.weight.grad is not None
    
    # 2. PixelShuffle & PixelUnshuffle inverse symmetry
    ps = nn.PixelShuffle(upscale_factor=2)
    pus = nn.PixelUnshuffle(downscale_factor=2)
    x_img = at.tensor(np.random.randn(2, 12, 4, 4).astype(np.float32), requires_grad=True)
    shuffled = ps(x_img)
    assert shuffled.shape == (2, 3, 8, 8)
    unshuffled = pus(shuffled)
    assert unshuffled.shape == (2, 12, 4, 4)
    np.testing.assert_allclose(unshuffled.numpy(), x_img.numpy())
    
    # 3. Upsample nearest
    up = nn.Upsample(scale_factor=2, mode='nearest')
    x_low = at.tensor(np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32), requires_grad=True)
    out_high = up(x_low)
    assert out_high.shape == (1, 1, 4, 4)
    np.testing.assert_allclose(out_high.numpy()[0, 0, 0:2, 0:2], np.full((2, 2), 1.0))

def test_fuzz_stack_chunk_split_unbind_and_autograd():
    """Fuzz test stack, chunk, split, unbind with Autograd."""
    import forge as at
    
    # 1. stack
    a = at.tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    b = at.tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
    c = at.tensor(np.array([5.0, 6.0], dtype=np.float32), requires_grad=True)
    st = at.stack([a, b, c], dim=0)
    assert st.shape == (3, 2)
    loss = (st * 2.0).sum()
    loss.backward()
    np.testing.assert_allclose(a.grad.numpy(), np.full(2, 2.0))
    np.testing.assert_allclose(b.grad.numpy(), np.full(2, 2.0))
    np.testing.assert_allclose(c.grad.numpy(), np.full(2, 2.0))
    
    # 2. chunk & SwiGLU-style splitting
    x = at.tensor(np.arange(12, dtype=np.float32).reshape(2, 6), requires_grad=True)
    c1, c2 = x.chunk(2, dim=-1)
    assert c1.shape == (2, 3)
    assert c2.shape == (2, 3)
    loss_chunk = (c1 * 3.0 + c2 * 5.0).sum()
    loss_chunk.backward()
    expected_grad = np.array([[3.0, 3.0, 3.0, 5.0, 5.0, 5.0], [3.0, 3.0, 3.0, 5.0, 5.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected_grad)
    
    # 3. split
    y = at.tensor(np.ones((4, 10), dtype=np.float32))
    s1, s2, s3 = y.split([2, 3, 5], dim=1)
    assert s1.shape == (4, 2)
    assert s2.shape == (4, 3)
    assert s3.shape == (4, 5)
    
    # 4. unbind
    z = at.tensor(np.arange(6, dtype=np.float32).reshape(3, 2))
    u0, u1, u2 = z.unbind(dim=0)
    assert u0.shape == (2,)
    assert u1.shape == (2,)
    assert u2.shape == (2,)
    np.testing.assert_allclose(u0.numpy(), np.array([0.0, 1.0]))

def test_fuzz_multinomial_randint_bernoulli_normal_sampling():
    """Fuzz test multinomial, randint, bernoulli, normal."""
    import forge as at
    
    # 1. multinomial 1D and 2D
    p_1d = at.tensor(np.array([0.1, 0.8, 0.1], dtype=np.float32))
    samples_1d = at.multinomial(p_1d, num_samples=10, replacement=True)
    assert samples_1d.shape == (10,)
    assert samples_1d.dtype == "int32"
    
    p_2d = at.tensor(np.array([[0.0, 1.0, 0.0], [0.99, 0.01, 0.0]], dtype=np.float32))
    samples_2d = p_2d.multinomial(num_samples=2, replacement=True)
    assert samples_2d.shape == (2, 2)
    assert samples_2d.numpy()[0, 0] == 1  # 1.0 prob is at index 1
    
    # 2. randint
    r = at.randint(10, 20, size=(3, 4))
    assert r.shape == (3, 4)
    assert (r.numpy() >= 10).all()
    assert (r.numpy() < 20).all()
    
    # 3. bernoulli
    b_p = at.tensor(np.full((5, 5), 0.5, dtype=np.float32))
    b = b_p.bernoulli()
    assert b.shape == (5, 5)
    assert set(np.unique(b.numpy())).issubset({0.0, 1.0})
    
    # 4. normal
    norm = at.normal(mean=5.0, std=0.01, size=(100,))
    assert norm.shape == (100,)
    np.testing.assert_allclose(norm.numpy().mean(), 5.0, atol=0.1)

def test_fuzz_parameter_and_module_containers():
    """Fuzz test nn.Parameter, Sequential, ModuleDict, ParameterList, ParameterDict."""
    import forge as at
    import forge.nn as nn
    
    # 1. Parameter auto requires_grad
    p = nn.Parameter(np.zeros((3, 3), dtype=np.float32))
    assert p.requires_grad == True
    assert isinstance(p, at.Tensor)
    
    # 2. Sequential append, slice, iter
    seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
    seq.append(nn.Linear(8, 2))
    assert len(seq) == 3
    assert len(list(iter(seq))) == 3
    sub_seq = seq[0:2]
    assert len(sub_seq) == 2
    
    x = at.tensor(np.ones((2, 4), dtype=np.float32), requires_grad=True)
    out = seq(x)
    assert out.shape == (2, 2)
    out.sum().backward()
    assert x.grad is not None
    
    # 3. ModuleDict
    mdict = nn.ModuleDict({
        'fc1': nn.Linear(4, 4),
        'act': nn.ReLU()
    })
    assert 'fc1' in mdict
    assert len(mdict.parameters()) > 0
    
    # 4. ParameterList
    plist = nn.ParameterList([nn.Parameter(np.ones(4)), nn.Parameter(np.ones(4))])
    plist.append(nn.Parameter(np.ones(4)))
    assert len(plist) == 3
    assert len(plist.parameters()) == 3
    
    # 5. ParameterDict
    pdict = nn.ParameterDict({'w': nn.Parameter(np.ones(4))})
    pdict['b'] = nn.Parameter(np.zeros(4))
    assert len(pdict.parameters()) == 2

def test_fuzz_roll_repeat_interleave_meshgrid_diag_outer_autograd():
    """Fuzz test roll, repeat_interleave, meshgrid, diag, diagonal, trace, outer with Autograd."""
    import forge as at
    
    # 1. roll & backward
    x = at.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)
    r = x.roll(shifts=(1, -1), dims=(0, 1))
    assert r.shape == (3, 4)
    loss = (r * 2.0).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad.numpy(), np.full((3, 4), 2.0))
    
    # 2. repeat_interleave & backward
    y = at.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    rep = y.repeat_interleave(3, dim=1)
    assert rep.shape == (2, 6)
    loss_rep = (rep * np.arange(12, dtype=np.float32).reshape(2, 6)).sum()
    loss_rep.backward()
    # Gradient sum of weights [0, 1, 2] -> 3, [3, 4, 5] -> 12, etc.
    np.testing.assert_allclose(y.grad.numpy(), np.array([[3.0, 12.0], [21.0, 30.0]], dtype=np.float32))
    
    # 3. meshgrid
    x_coords = at.tensor(np.array([1.0, 2.0], dtype=np.float32))
    y_coords = at.tensor(np.array([3.0, 4.0, 5.0], dtype=np.float32))
    grid_x, grid_y = at.meshgrid(x_coords, y_coords, indexing="ij")
    assert grid_x.shape == (2, 3)
    assert grid_y.shape == (2, 3)
    
    # 4. diag, diagonal, trace
    d_vec = at.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    mat = at.diag(d_vec)
    assert mat.shape == (3, 3)
    tr = mat.trace()
    tr.backward()
    np.testing.assert_allclose(d_vec.grad.numpy(), np.ones(3, dtype=np.float32))
    
    mat2 = at.tensor(np.arange(9, dtype=np.float32).reshape(3, 3))
    diag_vals = mat2.diagonal()
    np.testing.assert_allclose(diag_vals.numpy(), np.array([0.0, 4.0, 8.0]))
    
    # 5. outer product
    v1 = at.tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    v2 = at.tensor(np.array([3.0, 4.0, 5.0], dtype=np.float32), requires_grad=True)
    out_prod = at.outer(v1, v2)
    assert out_prod.shape == (2, 3)
    out_prod.sum().backward()
    np.testing.assert_allclose(v1.grad.numpy(), np.array([12.0, 12.0]))
    np.testing.assert_allclose(v2.grad.numpy(), np.array([3.0, 3.0, 3.0]))

def test_fuzz_masked_fill_index_select_masked_select_nonzero_take_along_dim():
    """Fuzz test masked_fill, index_select, masked_select, nonzero, take_along_dim with Autograd."""
    import forge as at
    
    # 1. masked_fill & backward
    scores = at.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    mask = at.tensor(np.array([[False, True], [True, False]], dtype=bool))
    filled = scores.masked_fill(mask, -1e9)
    assert filled.numpy()[0, 1] == -1e9
    assert filled.numpy()[1, 0] == -1e9
    loss = (filled * 2.0).sum()
    loss.backward()
    np.testing.assert_allclose(scores.grad.numpy(), np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    
    # 2. index_select & backward
    x = at.tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)
    idx = at.tensor(np.array([0, 2], dtype=np.int64))
    sel = x.index_select(dim=0, index=idx)
    assert sel.shape == (2, 4)
    loss_sel = (sel * 3.0).sum()
    loss_sel.backward()
    expected_grad = np.array([
        [3.0, 3.0, 3.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0, 3.0]
    ], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected_grad)
    
    # 3. masked_select & backward
    m_in = at.tensor(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32), requires_grad=True)
    bool_mask = at.tensor(np.array([True, False, True, False], dtype=bool))
    ms = m_in.masked_select(bool_mask)
    assert ms.shape == (2,)
    (ms * 5.0).sum().backward()
    np.testing.assert_allclose(m_in.grad.numpy(), np.array([5.0, 0.0, 5.0, 0.0], dtype=np.float32))
    
    # 4. nonzero
    nz_t = at.tensor(np.array([[0, 1], [2, 0]], dtype=np.int32))
    nz_indices = nz_t.nonzero()
    np.testing.assert_allclose(nz_indices.numpy(), np.array([[0, 1], [1, 0]], dtype=np.int64))
    
    # 5. take_along_dim
    t_in = at.tensor(np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32), requires_grad=True)
    gather_idx = at.tensor(np.array([[2, 0], [1, 2]], dtype=np.int64))
    t_out = t_in.take_along_dim(gather_idx, dim=1)
    assert t_out.shape == (2, 2)
    np.testing.assert_allclose(t_out.numpy(), np.array([[30.0, 10.0], [50.0, 60.0]]))

def test_fuzz_cumprod_unflatten_cdist_norm_atleast_xd():
    """Fuzz test cumprod, unflatten, cdist, norm, atleast_1d/2d/3d with Autograd."""
    import forge as at
    
    # 1. cumprod & backward
    a = at.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), requires_grad=True)
    cp = a.cumprod(dim=0)
    assert cp.shape == (4,)
    np.testing.assert_allclose(cp.numpy(), np.array([1.0, 2.0, 6.0, 24.0]))
    cp.sum().backward()
    # d(y1+y2+y3+y4)/da1 = 1 + a2 + a2*a3 + a2*a3*a4 = 1 + 2 + 6 + 24 = 33
    # d(y1+y2+y3+y4)/da2 = a1 + a1*a3 + a1*a3*a4 = 1 + 3 + 12 = 16
    # d(y1+y2+y3+y4)/da3 = a1*a2 + a1*a2*a4 = 2 + 8 = 10
    # d(y1+y2+y3+y4)/da4 = a1*a2*a3 = 6
    np.testing.assert_allclose(a.grad.numpy(), np.array([33.0, 16.0, 10.0, 6.0], dtype=np.float32))
    
    # 2. unflatten
    t_flat = at.tensor(np.zeros((2, 12, 5), dtype=np.float32))
    t_unflat = t_flat.unflatten(dim=1, sizes=(3, 4))
    assert t_unflat.shape == (2, 3, 4, 5)
    
    # 3. cdist
    x1 = at.tensor(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    x2 = at.tensor(np.array([[0.0, 0.0]], dtype=np.float32))
    dists = at.cdist(x1, x2, p=2.0)
    assert dists.shape == (2, 1)
    np.testing.assert_allclose(dists.numpy(), np.array([[0.0], [5.0]], dtype=np.float32))
    
    # 4. norm
    v = at.tensor(np.array([3.0, 4.0], dtype=np.float32), requires_grad=True)
    n = v.norm(p=2)
    np.testing.assert_allclose(n.numpy(), 5.0)
    n.backward()
    np.testing.assert_allclose(v.grad.numpy(), np.array([0.6, 0.8], dtype=np.float32))
    
    # 5. atleast_1d, 2d, 3d
    sc = at.tensor(5.0)
    assert at.atleast_1d(sc).shape == (1,)
    assert at.atleast_2d(sc).shape == (1, 1)
    assert at.atleast_3d(sc).shape == (1, 1, 1)

def test_fuzz_adamw_rmsprop_adagrad_linearlr_lambdalr_multisteplr():
    """Fuzz test AdamW, RMSprop, Adagrad, LinearLR, LambdaLR, MultiStepLR."""
    import forge as at
    from forge import optim
    
    # 1. AdamW
    w = at.tensor(np.array([10.0, 20.0], dtype=np.float32), requires_grad=True)
    opt_adamw = optim.AdamW([w], lr=0.1, weight_decay=0.01)
    w.grad = at.tensor(np.array([1.0, 1.0], dtype=np.float32))
    opt_adamw.step()
    # verify weight decay and step applied
    assert w.numpy()[0] < 10.0
    assert w.grad is None
    
    # 2. RMSprop
    w_rms = at.tensor(np.array([10.0, 20.0], dtype=np.float32), requires_grad=True)
    opt_rms = optim.RMSprop([w_rms], lr=0.1, alpha=0.9, momentum=0.1)
    w_rms.grad = at.tensor(np.array([2.0, 2.0], dtype=np.float32))
    opt_rms.step()
    assert w_rms.numpy()[0] < 10.0
    assert w_rms.grad is None
    
    # 3. Adagrad
    w_ada = at.tensor(np.array([10.0, 20.0], dtype=np.float32), requires_grad=True)
    opt_ada = optim.Adagrad([w_ada], lr=0.1)
    w_ada.grad = at.tensor(np.array([1.0, 1.0], dtype=np.float32))
    opt_ada.step()
    assert w_ada.numpy()[0] < 10.0
    assert w_ada.grad is None
    
    # 4. LinearLR
    opt = optim.SGD([w], lr=1.0)
    sched_lin = optim.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=4)
    # initial step called in __init__: lr = 1.0 * 0.2 = 0.2
    assert abs(opt.lr - 0.2) < 1e-5
    sched_lin.step() # epoch 1: factor = 0.2 + 0.8 * (1/4) = 0.4
    assert abs(opt.lr - 0.4) < 1e-5
    sched_lin.step() # epoch 2: factor = 0.2 + 0.8 * (2/4) = 0.6
    assert abs(opt.lr - 0.6) < 1e-5
    
    # 5. LambdaLR
    opt2 = optim.SGD([w], lr=1.0)
    sched_lam = optim.LambdaLR(opt2, lr_lambda=lambda epoch: 0.95 ** epoch)
    assert abs(opt2.lr - 1.0) < 1e-5
    sched_lam.step()
    assert abs(opt2.lr - 0.95) < 1e-5
    
    # 6. MultiStepLR
    opt3 = optim.SGD([w], lr=1.0)
    sched_ms = optim.MultiStepLR(opt3, milestones=[2, 4], gamma=0.5)
    sched_ms.step() # epoch 1
    assert abs(opt3.lr - 1.0) < 1e-5
    sched_ms.step() # epoch 2 (hit milestone!)
    assert abs(opt3.lr - 0.5) < 1e-5

def test_fuzz_einsum_kron_tensordot_nan_to_num():
    """Fuzz test einsum, kron, tensordot, nan_to_num with Autograd."""
    import forge as at
    
    # 1. einsum matrix multiplication & backward
    a = at.tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    b = at.tensor(np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), requires_grad=True)
    c = at.einsum("ij,jk->ik", a, b)
    assert c.shape == (2, 2)
    np.testing.assert_allclose(c.numpy(), np.array([[19.0, 22.0], [43.0, 50.0]]))
    c.sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), np.array([[11.0, 15.0], [11.0, 15.0]]))
    np.testing.assert_allclose(b.grad.numpy(), np.array([[4.0, 4.0], [6.0, 6.0]]))
    
    # 2. einsum batched trace & contraction
    x = at.tensor(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32))
    tr = at.einsum("bii->b", x)
    np.testing.assert_allclose(tr.numpy(), np.array([5.0]))
    
    # 3. kron & backward
    k1 = at.tensor(np.array([[1.0, 2.0]], dtype=np.float32), requires_grad=True)
    k2 = at.tensor(np.array([[3.0], [4.0]], dtype=np.float32), requires_grad=True)
    k_out = at.kron(k1, k2)
    assert k_out.shape == (2, 2)
    np.testing.assert_allclose(k_out.numpy(), np.array([[3.0, 6.0], [4.0, 8.0]]))
    
    # 4. tensordot
    td1 = at.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    td2 = at.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
    td_out = at.tensordot(td1, td2, dims=1)
    assert td_out.shape == (2, 4)
    
    # 5. nan_to_num & backward
    with_nan = at.tensor(np.array([1.0, float('nan'), float('inf'), float('-inf')], dtype=np.float32), requires_grad=True)
    cleaned = with_nan.nan_to_num(nan=0.0, posinf=999.0, neginf=-999.0)
    np.testing.assert_allclose(cleaned.numpy(), np.array([1.0, 0.0, 999.0, -999.0]))
    (cleaned * 2.0).sum().backward()
    np.testing.assert_allclose(with_nan.grad.numpy(), np.array([2.0, 0.0, 0.0, 0.0]))

def test_fuzz_conv_transpose2d_affine_grid_grid_sample():
    """Fuzz test ConvTranspose2d, conv_transpose2d, affine_grid, grid_sample with Autograd."""
    import forge as at
    from forge import nn
    
    # 1. ConvTranspose2d Module & backward
    deconv = nn.ConvTranspose2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
    x = at.tensor(np.ones((1, 2, 4, 4), dtype=np.float32), requires_grad=True)
    out = deconv(x)
    # H_out = (4 - 1)*2 - 2*1 + 3 = 6 - 2 + 3 = 7
    assert out.shape == (1, 3, 7, 7)
    out.sum().backward()
    assert x.grad is not None
    assert deconv.weight.grad is not None
    assert deconv.bias.grad is not None
    
    # 2. affine_grid & grid_sample
    # Identity affine matrix [ [1, 0, 0], [0, 1, 0] ]
    theta = at.tensor(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32))
    grid = at.affine_grid(theta, size=[1, 1, 4, 4], align_corners=True)
    assert grid.shape == (1, 4, 4, 2)
    
    img = at.tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4), requires_grad=True)
    sampled = at.grid_sample(img, grid, align_corners=True)
    assert sampled.shape == (1, 1, 4, 4)
    # Sampling with identity grid should recover the original image
    np.testing.assert_allclose(sampled.numpy(), img.numpy(), atol=1e-4)
    
    sampled.sum().backward()
    assert img.grad is not None
    np.testing.assert_allclose(img.grad.numpy(), np.ones((1, 1, 4, 4), dtype=np.float32), atol=1e-4)

def test_fuzz_ranking_losses_and_label_smoothing():
    """Fuzz test TripletMarginLoss, CosineEmbeddingLoss, MarginRankingLoss and CrossEntropy with label_smoothing."""
    import forge as at
    from forge import nn
    import forge.functional as F
    
    # 1. TripletMarginLoss & backward
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2.0)
    anchor = at.tensor(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), requires_grad=True)
    positive = at.tensor(np.array([[0.1, 0.0], [1.1, 1.0]], dtype=np.float32), requires_grad=True)
    negative = at.tensor(np.array([[2.0, 2.0], [0.0, 0.0]], dtype=np.float32), requires_grad=True)
    loss = triplet_loss(anchor, positive, negative)
    assert loss.shape == ()
    loss.backward()
    assert anchor.grad is not None
    assert positive.grad is not None
    assert negative.grad is not None
    
    # 2. CosineEmbeddingLoss & backward
    cos_loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
    x1 = at.tensor(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), requires_grad=True)
    x2 = at.tensor(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32), requires_grad=True)
    target = at.tensor(np.array([1.0, -1.0], dtype=np.float32))
    l_cos = cos_loss_fn(x1, x2, target)
    assert l_cos.shape == ()
    l_cos.backward()
    assert x1.grad is not None
    assert x2.grad is not None
    
    # 3. MarginRankingLoss & backward
    rank_loss_fn = nn.MarginRankingLoss(margin=0.5)
    r1 = at.tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    r2 = at.tensor(np.array([0.5, 3.0], dtype=np.float32), requires_grad=True)
    t_rank = at.tensor(np.array([1.0, -1.0], dtype=np.float32))
    l_rank = rank_loss_fn(r1, r2, t_rank)
    assert l_rank.shape == ()
    l_rank.backward()
    assert r1.grad is not None
    assert r2.grad is not None
    
    # 4. CrossEntropyLoss with label_smoothing
    ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
    logits = at.tensor(np.array([[2.0, 1.0, 0.1], [0.5, 3.0, 1.0]], dtype=np.float32), requires_grad=True)
    ce_targets = at.tensor(np.array([0, 1], dtype=np.float32))
    l_ce = ce_smooth(logits, ce_targets)
    assert l_ce.shape == ()
    l_ce.backward()
    assert logits.grad is not None

def test_fuzz_linalg_suite():
    """Fuzz test forge.linalg module: inv, det, pinv, cholesky, qr, svd, eigh, matrix_rank, norm with Autograd."""
    import forge as at
    from forge import linalg
    
    # 1. linalg.inv & backward
    A_np = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    A = at.tensor(A_np, requires_grad=True)
    A_inv = linalg.inv(A)
    assert A_inv.shape == (2, 2)
    np.testing.assert_allclose(A_inv.numpy(), np.linalg.inv(A_np), atol=1e-4)
    A_inv.sum().backward()
    assert A.grad is not None
    
    # 2. linalg.det & backward
    B = at.tensor(np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32), requires_grad=True)
    d = linalg.det(B)
    assert d.shape == ()
    np.testing.assert_allclose(d.numpy(), 3.0, atol=1e-4)
    d.backward()
    assert B.grad is not None
    
    # 3. linalg.cholesky
    L = linalg.cholesky(A)
    assert L.shape == (2, 2)
    np.testing.assert_allclose(L.numpy() @ L.numpy().T, A_np, atol=1e-4)
    
    # 4. linalg.qr
    Q, R = linalg.qr(A)
    assert Q.shape == (2, 2)
    assert R.shape == (2, 2)
    np.testing.assert_allclose(Q.numpy() @ R.numpy(), A_np, atol=1e-4)
    
    # 5. linalg.svd
    U, S, Vh = linalg.svd(A)
    assert U.shape == (2, 2)
    assert S.shape == (2,)
    assert Vh.shape == (2, 2)
    reconstructed = U.numpy() @ np.diag(S.numpy()) @ Vh.numpy()
    np.testing.assert_allclose(reconstructed, A_np, atol=1e-4)
    
    # 6. linalg.pinv & matrix_rank & norm
    pinv_A = linalg.pinv(A)
    assert pinv_A.shape == (2, 2)
    np.testing.assert_allclose(pinv_A.numpy(), np.linalg.pinv(A_np), atol=1e-4)
    
    rank = linalg.matrix_rank(A)
    assert int(rank.numpy()) == 2
    
    n = linalg.norm(A)
    np.testing.assert_allclose(n.numpy(), np.linalg.norm(A_np), atol=1e-4)



























