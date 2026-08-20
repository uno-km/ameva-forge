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

