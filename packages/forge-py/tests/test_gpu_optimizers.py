"""
test_gpu_optimizers.py - WebGPU Native Adam & Momentum SGD Optimizer End-to-End Execution Tests
"""
import pytest
import numpy as np
import json
from unittest.mock import patch
import forge as torch
from forge.optim import SGD, Adam
from forge.errors import AMEVAForgeDeviceError


@pytest.mark.asyncio
async def test_gpu_momentum_sgd_step_async_real_execution():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype="float32", device="gpu")
    p._handle = "param_handle_1"
    p.grad = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype="float32", device="gpu")
    p.grad._handle = "grad_handle_1"

    opt = SGD([p], lr=0.01, momentum=0.9)

    captured_instructions = []

    async def mock_execute(instructions, inputs):
        nonlocal captured_instructions
        parsed = json.loads(instructions)
        captured_instructions = parsed
        # Return in-place param handle for the sgd_momentum_step op node
        step_nodes = [node["id"] for node in parsed if node.get("op") == "sgd_momentum_step"]
        # Velocity upload node
        upload_nodes = [node["id"] for node in parsed if node.get("op") == "upload"]
        res = {}
        for uid in upload_nodes:
            res[str(uid)] = "vel_handle_allocated"
        for sid in step_nodes:
            res[str(sid)] = "param_handle_1"  # in-place contract
        return res

    with patch("forge.bridge.js_execute_graph", side_effect=mock_execute):
        # 💥 DIRECT PHYSICAL EXECUTION OF opt.step_async()
        await opt.step_async()

    assert len(captured_instructions) > 0
    step_inst = [i for i in captured_instructions if i.get("op") == "sgd_momentum_step"]
    assert len(step_inst) == 1
    assert step_inst[0]["in"][0] == 1  # param_id
    assert step_inst[0]["in"][1] == 2  # grad_id
    assert step_inst[0]["params"] == [0.01, 0.9]
    assert opt.velocity[0]._handle == "vel_handle_allocated"
    assert p._handle == "param_handle_1"


@pytest.mark.asyncio
async def test_gpu_adam_step_async_real_execution():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype="float32", device="gpu")
    p._handle = "param_handle_adam"
    p.grad = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype="float32", device="gpu")
    p.grad._handle = "grad_handle_adam"

    opt = Adam([p], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    captured_instructions = []

    async def mock_execute(instructions, inputs):
        nonlocal captured_instructions
        parsed = json.loads(instructions)
        captured_instructions = parsed
        step_nodes = [node["id"] for node in parsed if node.get("op") == "adam_step"]
        upload_nodes = [node["id"] for node in parsed if node.get("op") == "upload"]
        res = {}
        for uid in upload_nodes:
            res[str(uid)] = f"handle_upload_{uid}"
        for sid in step_nodes:
            res[str(sid)] = "param_handle_adam"  # in-place contract
        return res

    with patch("forge.bridge.js_execute_graph", side_effect=mock_execute):
        # 💥 DIRECT PHYSICAL EXECUTION OF opt.step_async()
        await opt.step_async()

    assert len(captured_instructions) > 0
    step_inst = [i for i in captured_instructions if i.get("op") == "adam_step"]
    assert len(step_inst) == 1
    assert step_inst[0]["in"][0] == 1  # param_id
    assert step_inst[0]["in"][1] == 2  # grad_id
    assert step_inst[0]["params"][0] == 0.001  # lr
    assert step_inst[0]["params"][1] == 0.9    # beta1
    assert step_inst[0]["params"][2] == 0.999  # beta2
    assert step_inst[0]["params"][3] == 1e-8   # eps
    assert opt.m[0]._handle is not None
    assert opt.v[0]._handle is not None
    assert p._handle == "param_handle_adam"
    assert p.grad is None


@pytest.mark.asyncio
async def test_gpu_adam_contract_violation_raises():
    p = torch.tensor([1.0, 2.0], dtype="float32", device="gpu")
    p._handle = "orig_handle"
    p.grad = torch.tensor([0.1, 0.2], dtype="float32", device="gpu")
    p.grad._handle = "orig_grad_handle"

    opt = Adam([p], lr=0.001)

    async def mock_execute_violation(instructions, inputs):
        parsed = json.loads(instructions)
        res = {}
        for node in parsed:
            res[str(node["id"])] = "different_corrupted_handle"
        return res

    with patch("forge.bridge.js_execute_graph", side_effect=mock_execute_violation):
        with pytest.raises(AMEVAForgeDeviceError, match="Adam contract violation"):
            await opt.step_async()
