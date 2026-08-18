import asyncio
import json
import numpy as np
import pytest

from forge.optim import SGD
from forge.tensor import Tensor
from forge.errors import AMEVAForgeDeviceError


def test_sync_sgd_rejects_gpu_parameter_with_clear_message():
    p = Tensor(
        shape=(2,), dtype="float32", device="gpu",
        requires_grad=True, handle="param-handle"
    )
    p.grad = Tensor(
        shape=(2,), dtype="float32", device="gpu",
        requires_grad=False, handle="grad-handle"
    )

    with pytest.raises(AMEVAForgeDeviceError, match="step_async"):
        SGD([p], lr=0.1).step()


def test_cpu_step_async_updates_parameter():
    p = Tensor(
        shape=(2,), dtype="float32", device="cpu",
        requires_grad=True,
        data=np.array([1.0, 2.0], dtype=np.float32),
    )
    p.grad = Tensor(
        shape=(2,), dtype="float32", device="cpu",
        data=np.array([0.5, -1.0], dtype=np.float32),
    )

    asyncio.run(SGD([p], lr=0.1).step_async())
    np.testing.assert_allclose(p.numpy(), [0.95, 2.1], rtol=0, atol=1e-6)
    assert p.grad is None


def test_gpu_step_async_builds_in_place_axpy(monkeypatch):
    p = Tensor(
        shape=(2,), dtype="float32", device="gpu",
        requires_grad=True, handle="param-handle"
    )
    p.grad = Tensor(
        shape=(2,), dtype="float32", device="gpu",
        handle="grad-handle"
    )

    captured = {}

    async def fake_execute(instructions, inputs):
        nodes = json.loads(instructions)
        captured["nodes"] = nodes
        axpy = next(node for node in nodes if node["op"] == "axpy")
        return {str(axpy["id"]): "param-handle"}

    monkeypatch.setattr("forge.bridge.js_execute_graph", fake_execute)

    asyncio.run(SGD([p], lr=0.1).step_async())

    axpy = next(node for node in captured["nodes"] if node["op"] == "axpy")
    assert axpy["params"] == [2, 0.1]
    assert p._handle == "param-handle"
    assert p.grad is None
