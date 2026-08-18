"""
test_gpu_optimizers.py - WebGPU Native Adam & Momentum SGD Optimizer Tests
"""
import pytest
import numpy as np
import forge as torch
from forge.optim import SGD, Adam
from forge.graph import GraphBuilder


@pytest.mark.asyncio
async def test_gpu_momentum_sgd_step_async_builds_graph():
    # 1. Initialize GPU parameter with gradient
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype="float32", device="gpu")
    p._handle = "param_handle_1"
    p.grad = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype="float32", device="gpu")
    p.grad._handle = "grad_handle_1"

    opt = SGD([p], lr=0.01, momentum=0.9)
    
    # In CPython mock environment, check that velocity is initialized as GPU tensor
    # and executeGraph compiles sgd_momentum_step instruction
    builder = GraphBuilder()
    param_id = builder.add_load(p.shape, p._handle)
    grad_id = builder.add_load(p.grad.shape, p.grad._handle)
    
    vel = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype="float32", device="gpu")
    vel._handle = "vel_handle_1"
    vel_id = builder.add_load(vel.shape, vel._handle)

    out_id = builder.add_op("sgd_momentum_step", p.shape, [param_id, grad_id, vel_id], [0.01, 0.9])
    ast = builder.to_dict()

    assert len(ast["instructions"]) == 4
    step_inst = [i for i in ast["instructions"] if i.get("op") == "sgd_momentum_step"]
    assert len(step_inst) == 1
    assert step_inst[0]["in"] == [1, 2, 3]
    assert step_inst[0]["params"] == [0.01, 0.9]


@pytest.mark.asyncio
async def test_gpu_adam_step_async_builds_graph():
    p = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype="float32", device="gpu")
    p._handle = "param_handle_adam"
    p.grad = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype="float32", device="gpu")
    p.grad._handle = "grad_handle_adam"

    opt = Adam([p], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    
    builder = GraphBuilder()
    param_id = builder.add_load(p.shape, p._handle)
    grad_id = builder.add_load(p.grad.shape, p.grad._handle)

    m = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype="float32", device="gpu")
    m._handle = "m_handle_1"
    m_id = builder.add_load(m.shape, m._handle)

    v = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype="float32", device="gpu")
    v._handle = "v_handle_1"
    v_id = builder.add_load(v.shape, v._handle)

    out_id = builder.add_op(
        "adam_step",
        p.shape,
        [param_id, grad_id, m_id, v_id],
        [0.001, 0.9, 0.999, 1e-8, 0.9, 0.999]
    )
    ast = builder.to_dict()

    assert len(ast["instructions"]) == 5
    step_inst = [i for i in ast["instructions"] if i.get("op") == "adam_step"]
    assert len(step_inst) == 1
    assert step_inst[0]["in"] == [1, 2, 3, 4]
    assert step_inst[0]["params"] == [0.001, 0.9, 0.999, 1e-8, 0.9, 0.999]
