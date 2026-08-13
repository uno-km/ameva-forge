import numpy as np
import pytest
import forge as at
from forge.tensor import Tensor

def test_slice_row_index():
    x = at.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, dtype='float32')
    y = x[0]
    assert y.shape == (3,)
    np.testing.assert_array_equal(y.numpy(), [1, 2, 3])
    
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[1, 1, 1], [0, 0, 0]])

def test_slice_col_index():
    x = at.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, dtype='float32')
    y = x[:, 1]
    assert y.shape == (2,)
    np.testing.assert_array_equal(y.numpy(), [2, 5])
    
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[0, 1, 0], [0, 1, 0]])

def test_slice_range():
    x = at.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, dtype='float32')
    y = x[0:2, 1:]
    assert y.shape == (2, 2)
    np.testing.assert_array_equal(y.numpy(), [[2, 3], [5, 6]])
    
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[0, 1, 1], [0, 1, 1], [0, 0, 0]])

def test_slice_negative_index():
    x = at.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, dtype='float32')
    y = x[-1]
    assert y.shape == (3,)
    np.testing.assert_array_equal(y.numpy(), [4, 5, 6])
    
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[0, 0, 0], [1, 1, 1]])

def test_slice_tuple_index():
    x = at.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True, dtype='float32')
    y = x[(0, 2)]
    assert y.shape == tuple()
    assert y.numpy() == 3
    
    y.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[0, 0, 1], [0, 0, 0]])

def test_slice_repeated_index_accumulation():
    x = at.tensor([1, 2, 3, 4], requires_grad=True, dtype='float32')
    y = x[[0, 1, 0, 2]]
    assert y.shape == (4,)
    np.testing.assert_array_equal(y.numpy(), [1, 2, 1, 3])
    
    y.sum().backward()
    # 0번 인덱스가 두 번 선택되었으므로 2가 누적되어야 함
    np.testing.assert_array_equal(x.grad.numpy(), [2, 1, 1, 0])

def test_slice_disposed_tensor():
    x = at.tensor([1, 2, 3], requires_grad=True, dtype='float32')
    x.dispose()
    with pytest.raises(Exception):
        y = x[0:2]

def test_scatter_regression():
    x = at.tensor([[1, 2], [3, 4]], requires_grad=True, dtype='float32')
    index = at.tensor([[0, 1], [1, 0]], dtype='int32')
    src = at.tensor([[10, 20], [30, 40]], requires_grad=True, dtype='float32')
    
    y = at.scatter(x, 1, index, src)
    np.testing.assert_array_equal(y.numpy(), [[10, 20], [40, 30]])
    
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [[0, 0], [0, 0]])
    np.testing.assert_array_equal(src.grad.numpy(), [[1, 1], [1, 1]])

def test_conv2d_regression():
    import forge.nn as nn
    x = at.tensor(np.ones((1, 1, 3, 3), dtype=np.float32), requires_grad=True)
    conv = nn.Conv2d(1, 1, 2, bias=False)
    conv.weight = at.tensor(np.ones((1, 1, 2, 2), dtype=np.float32), requires_grad=True)
    y = conv(x)
    assert y.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(y.numpy(), np.full((1, 1, 2, 2), 4.0))
    y.sum().backward()
    
    expected_grad = np.array([
        [[[1, 2, 1],
          [2, 4, 2],
          [1, 2, 1]]]]
    )
    np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

if __name__ == '__main__':
    pytest.main([__file__])
