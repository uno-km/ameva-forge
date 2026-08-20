import pytest
import numpy as np
from forge.ops import tensor, permute, max_op, max_axis, dropout, gather, scatter, AMEVAForgeShapeError


def test_permute_validation():
    x = tensor(np.random.randn(2, 3, 4))
    
    # Negative dims
    y = permute(x, (2, -2, -3)) # (2, 1, 0)
    assert y.shape == (4, 3, 2)
    
    # Invalid length
    with pytest.raises(AMEVAForgeShapeError):
        permute(x, (1, 0))
        
    # Duplicate dims
    with pytest.raises(ValueError):
        permute(x, (1, 1, 2))
        
    # Out of range
    with pytest.raises(IndexError):
        permute(x, (0, 1, 3))

def test_dropout_validation():
    x = tensor(np.random.randn(2, 2))
    with pytest.raises(ValueError):
        dropout(x, p=1.0)
    with pytest.raises(ValueError):
        dropout(x, p=-0.1)

def test_max_tie_breaking():
    x = tensor(np.array([[1.0, 3.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float32), requires_grad=True)
    
    # max_op
    y = max_op(x)
    y.backward()
    # 3 occurrences of 3.0. Gradient should be 1/3 each.
    expected_grad = np.array([[0, 1/3, 1/3], [1/3, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-5)
    
def test_max_axis_tie_breaking():
    x = tensor(np.array([[1.0, 3.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float32), requires_grad=True)
    y = max_axis(x, axis=1) # [3.0, 3.0]
    y.backward(tensor(np.array([2.0, 2.0], dtype=np.float32)))
    # Row 0: two 3.0s, so 2.0/2 = 1.0 each. Row 1: one 3.0, so 2.0/1 = 2.0
    expected_grad = np.array([[0, 1.0, 1.0], [2.0, 0, 0]], dtype=np.float32)
    np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-5)

def test_gather_scatter_grad_parents():
    # Test autograd return logic for index tensor
    src = tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    index = tensor(np.array([[0, 0], [1, 0]], dtype=np.int32))
    
    # Gather
    y = gather(src, 1, index)
    # y._grad_parents should be (src, index)
    assert len(y._grad_parents) == 2
    assert y._grad_parents[0] is src
    assert y._grad_parents[1] is index
    
    y.backward(tensor(np.ones_like(y.numpy(), dtype=np.float32)))
    assert src.grad is not None
    assert index.grad is None  # Since index shouldn't receive gradients, but it's explicitly None
    
    # Scatter
    src2 = tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
    target = tensor(np.zeros((2, 3), dtype=np.float32), requires_grad=True)
    index2 = tensor(np.array([[0, 1], [2, 0]], dtype=np.int32))
    
    z = scatter(target, 1, index2, src2)
    # z._grad_parents should be (target, index2, src2)
    assert len(z._grad_parents) == 3
    assert z._grad_parents[0] is target
    assert z._grad_parents[1] is index2
    assert z._grad_parents[2] is src2
    
    z.backward(tensor(np.ones_like(z.numpy(), dtype=np.float32)))
    assert src2.grad is not None
    assert index2.grad is None
    assert target.grad is not None


def test_bmm_unbroadcasting_backward():
    """Verify BMM unbroadcasting during backward pass."""
    from forge.ops import bmm
    a = tensor(np.random.randn(1, 4, 3).astype(np.float32), requires_grad=True)
    b = tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    
    c = bmm(a, b)
    assert c.shape == (2, 4, 4)
    
    loss = c.sum()
    loss.backward()
    
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (1, 4, 3)
    assert b.grad.shape == (2, 3, 4)


def test_integer_tensor_requires_grad_validation():
    """Verify that integer tensors cannot have requires_grad=True."""
    from forge.errors import AMEVAForgeValidationError
    with pytest.raises(AMEVAForgeValidationError):
        tensor([1, 2, 3], dtype="int32", requires_grad=True)
    with pytest.raises(AMEVAForgeValidationError):
        tensor([1, 2, 3], dtype="int64", requires_grad=True)


def test_multidimensional_layer_norm():
    """Verify multi-dimensional normalized_shape in LayerNorm."""
    from forge.functional import layer_norm
    x = tensor(np.random.randn(2, 3, 4, 5).astype(np.float32), requires_grad=True)
    norm_shape = (4, 5)
    
    y = layer_norm(x, norm_shape)
    assert y.shape == (2, 3, 4, 5)
    
    # Check that mean over last 2 dims is approximately 0 and var is approximately 1
    y_np = y.numpy()
    mean = np.mean(y_np, axis=(-2, -1))
    std = np.std(y_np, axis=(-2, -1))
    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-5)
    np.testing.assert_allclose(std, np.ones_like(std), atol=1e-3)


def test_all_masked_softmax_nan_defense():
    """Verify that an all -inf row does not produce NaN in Softmax or LogSoftmax."""
    from forge.functional import softmax, log_softmax
    logits = np.array([
        [1.0, 2.0, 3.0],
        [-np.inf, -np.inf, -np.inf]  # All masked row
    ], dtype=np.float32)
    
    t = tensor(logits)
    sm = softmax(t, axis=-1)
    sm_data = sm.numpy()
    
    assert not np.isnan(sm_data).any(), "Softmax output contains NaN!"
    assert np.all(sm_data[1] == 0.0), "All-masked row in Softmax should yield 0.0"
    
    lsm = log_softmax(t, axis=-1)
    lsm_data = lsm.numpy()
    assert not np.isnan(lsm_data).any(), "LogSoftmax output contains NaN!"

