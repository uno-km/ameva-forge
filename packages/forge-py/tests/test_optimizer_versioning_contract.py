import asyncio
import numpy as np
import pytest

from forge import tensor, nn
from forge.optim import SGD
from forge.tensor import Tensor
from forge.errors import AMEVAForgeDeviceError


def test_step_async_rejects_mixed_devices():
    cpu = Tensor(
        shape=(1,), dtype="float32", device="cpu",
        data=np.array([1.0], dtype=np.float32), requires_grad=True,
    )
    gpu = Tensor(
        shape=(1,), dtype="float32", device="gpu",
        handle="gpu-param", requires_grad=True,
    )
    cpu.grad = Tensor(
        shape=(1,), dtype="float32", device="cpu",
        data=np.array([1.0], dtype=np.float32),
    )
    gpu.grad = Tensor(
        shape=(1,), dtype="float32", device="gpu",
        handle="gpu-grad",
    )

    with pytest.raises(AMEVAForgeDeviceError, match="Mixed CPU/GPU"):
        asyncio.run(SGD([cpu, gpu], lr=0.1).step_async())

    np.testing.assert_allclose(cpu.numpy(), [1.0])
    assert cpu.grad is not None
    assert gpu.grad is not None


def test_optimizer_step_increments_tensor_version():
    p = Tensor(
        shape=(2,), dtype="float32", device="cpu",
        requires_grad=True,
        data=np.array([1.0, 2.0], dtype=np.float32),
    )
    p.grad = Tensor(
        shape=(2,), dtype="float32", device="cpu",
        data=np.array([0.5, 0.5], dtype=np.float32),
    )
    assert p._version == 0

    SGD([p], lr=0.1).step()
    assert p._version == 1


def test_backward_rejects_stale_parameter_version():
    x = tensor([[1.0]], requires_grad=True)
    w = tensor([[2.0]], requires_grad=True)
    loss = (x @ w).sum()

    w._data -= 0.1
    w._version += 1

    with pytest.raises(RuntimeError, match=r"modified.*inplace"):
        loss.backward()


def test_module_parameter_replacement_updates_registry():
    layer = nn.Linear(2, 1)
    old_weight = layer.weight
    new_weight = old_weight.to("gpu")
    layer.weight = new_weight

    params = layer.parameters()
    assert new_weight in params
    assert old_weight not in params
