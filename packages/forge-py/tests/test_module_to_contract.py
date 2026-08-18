import pytest
from forge import nn, optim
from forge.errors import AMEVAForgeDeviceError


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1, bias=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(2, 4, bias=True),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Linear(4, 1, bias=True)
        )

    def forward(self, x):
        return self.block2(self.block1(x))


def test_module_to_gpu_replaces_registered_parameters():
    model = MLP()
    previous = list(model.parameters())
    returned = model.to("gpu")
    current = list(model.parameters())
    assert returned is model
    assert all(parameter.device == "gpu" for parameter in current)
    assert all(
        old_parameter is not new_parameter
        for old_parameter, new_parameter in zip(previous, current)
    )


def test_module_to_rejects_unknown_device():
    model = MLP()
    with pytest.raises(
        AMEVAForgeDeviceError,
        match="Unsupported device",
    ):
        model.to("cuda")


def test_module_to_nested_submodules_and_bias():
    model = NestedModel()
    # verify initial on CPU
    for p in model.parameters():
        assert p.device == "cpu"
        assert p.requires_grad is True

    model.to("gpu")

    # verify nested submodules & bias are all on GPU with requires_grad preserved
    params = list(model.parameters())
    assert len(params) == 4  # fc1.weight, fc1.bias, fc2.weight, fc2.bias
    assert all(p.device == "gpu" for p in params)
    assert all(p.requires_grad is True for p in params)
    assert model.block1[0].weight.device == "gpu"
    assert model.block1[0].bias.device == "gpu"
    assert model.block2[0].weight.device == "gpu"
    assert model.block2[0].bias.device == "gpu"


def test_module_to_gpu_cpu_roundtrip():
    model = MLP()
    model.to("gpu")
    assert all(p.device == "gpu" for p in model.parameters())
    assert all(p.requires_grad is True for p in model.parameters())

    model.to("cpu")
    assert all(p.device == "cpu" for p in model.parameters())
    assert all(p.requires_grad is True for p in model.parameters())


def test_module_to_repeated_gpu_idempotent():
    model = MLP()
    model.to("gpu")
    first_gpu_params = list(model.parameters())
    model.to("gpu")
    second_gpu_params = list(model.parameters())
    assert all(p.device == "gpu" for p in second_gpu_params)
    assert len(first_gpu_params) == len(second_gpu_params)


def test_module_to_optimizer_parameter_sync():
    model = MLP()
    model.to("gpu")
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model_param_ids = [id(p) for p in model.parameters()]
    optim_param_ids = [id(p) for p in optimizer.params]
    assert model_param_ids == optim_param_ids
    assert all(p.device == "gpu" for p in optimizer.params)
