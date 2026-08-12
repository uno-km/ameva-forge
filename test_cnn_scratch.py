import sys
import os
import numpy as np

# Adjust path to find forge package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages', 'forge-py', 'src'))

import forge as at
import forge.nn as nn

print("Testing Phase B (CNN)...")

# Dummy input: N=2, C=3, H=32, W=32
x = at.tensor(np.random.randn(2, 3, 32, 32).astype(np.float32), requires_grad=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = SimpleCNN()
model.train()

# Forward pass
out = model(x)
print("Output shape:", out.shape)
assert out.shape == (2, 10)

# Dummy loss (sum)
loss = out.sum()
print("Loss computed:", float(loss.numpy()))

# Backward pass
print("Running backward...")
loss.backward()

print("Conv1 weight grad shape:", model.conv1.weight.grad.shape)
print("Conv1 bias grad shape:", model.conv1.bias.grad.shape)
print("FC weight grad shape:", model.fc.weight.grad.shape)

print("SUCCESS: Phase B CNN modules are working!")
