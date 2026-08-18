import numpy as np
import pytest
from forge import nn, tensor


def finite_difference(fn, value: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    gradient = np.zeros_like(value, dtype=np.float32)
    for index in np.ndindex(value.shape):
        plus = value.copy()
        minus = value.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (fn(plus) - fn(minus)) / (2.0 * epsilon)
    return gradient


def test_finite_difference_linear_mse():
    np.random.seed(42)
    x_val = np.random.randn(2, 3).astype(np.float32)
    target_val = np.random.randn(2, 2).astype(np.float32)

    layer = nn.Linear(3, 2, bias=True)
    loss_fn = nn.MSELoss()

    w_val = layer.weight.numpy().copy()

    # Analytical
    x = tensor(x_val)
    target = tensor(target_val)
    out = layer(x)
    loss = loss_fn(out, target)
    loss.backward()

    analytical_grad = layer.weight.grad.numpy()

    # Numerical with robust try/finally weight restoration
    def loss_eval(w_new):
        old_w = layer.weight._data
        try:
            layer.weight._data = w_new
            out_num = layer(tensor(x_val))
            l_num = loss_fn(out_num, tensor(target_val))
            return float(l_num.numpy())
        finally:
            layer.weight._data = old_w

    numerical_grad = finite_difference(loss_eval, w_val)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_grad,
        rtol=1e-3,
        atol=1e-4,
    )


def test_finite_difference_linear_relu_mse():
    np.random.seed(123)
    x_val = np.random.randn(3, 4).astype(np.float32)
    target_val = np.random.randn(3, 2).astype(np.float32)

    layer = nn.Linear(4, 2, bias=True)
    relu = nn.ReLU()
    loss_fn = nn.MSELoss()

    w_val = layer.weight.numpy().copy()

    # Analytical
    x = tensor(x_val)
    target = tensor(target_val)
    out = relu(layer(x))
    loss = loss_fn(out, target)
    loss.backward()

    analytical_grad = layer.weight.grad.numpy()

    # Numerical with robust try/finally weight restoration
    def loss_eval(w_new):
        old_w = layer.weight._data
        try:
            layer.weight._data = w_new
            out_num = relu(layer(tensor(x_val)))
            l_num = loss_fn(out_num, tensor(target_val))
            return float(l_num.numpy())
        finally:
            layer.weight._data = old_w

    numerical_grad = finite_difference(loss_eval, w_val)

    np.testing.assert_allclose(
        analytical_grad,
        numerical_grad,
        rtol=1e-3,
        atol=1e-4,
    )


def test_finite_difference_2layer_mlp_mse():
    np.random.seed(456)
    x_val = np.random.randn(4, 3).astype(np.float32)
    target_val = np.random.randn(4, 1).astype(np.float32)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 4, bias=True)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 1, bias=True)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = MLP()
    loss_fn = nn.MSELoss()

    w1_val = model.fc1.weight.numpy().copy()
    w2_val = model.fc2.weight.numpy().copy()

    # Analytical
    out = model(tensor(x_val))
    loss = loss_fn(out, tensor(target_val))
    loss.backward()

    analytical_grad_w1 = model.fc1.weight.grad.numpy()
    analytical_grad_w2 = model.fc2.weight.grad.numpy()

    # Numerical w1 with robust try/finally weight restoration
    def loss_eval_w1(w1_new):
        old = model.fc1.weight._data
        try:
            model.fc1.weight._data = w1_new
            out_num = model(tensor(x_val))
            l_num = loss_fn(out_num, tensor(target_val))
            return float(l_num.numpy())
        finally:
            model.fc1.weight._data = old

    # Numerical w2 with robust try/finally weight restoration
    def loss_eval_w2(w2_new):
        old = model.fc2.weight._data
        try:
            model.fc2.weight._data = w2_new
            out_num = model(tensor(x_val))
            l_num = loss_fn(out_num, tensor(target_val))
            return float(l_num.numpy())
        finally:
            model.fc2.weight._data = old

    num_grad_w1 = finite_difference(loss_eval_w1, w1_val)
    num_grad_w2 = finite_difference(loss_eval_w2, w2_val)

    np.testing.assert_allclose(
        analytical_grad_w1,
        num_grad_w1,
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        analytical_grad_w2,
        num_grad_w2,
        rtol=1e-3,
        atol=1e-4,
    )
