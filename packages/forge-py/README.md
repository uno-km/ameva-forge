# AMEVA-Forge v2.0.0

**WebGPU-accelerated tensor computation library with autograd for deep learning training.**

AMEVA-Forge provides a PyTorch-like API for tensor operations with automatic differentiation and GPU acceleration via WebGPU. Version 2.0 adds full deep learning training capabilities including backward pass, optimizers, loss functions, and neural network modules.

---

## Features

### Core Tensor Operations
- **Arithmetic**: add, sub, mul, div, neg (with broadcasting)
- **Matrix**: matmul, transpose
- **Reduction**: sum, mean, max (N-Pass GPU reduction)
- **Element-wise Math**: exp, log, reshape, view
- **Activations**: relu, sigmoid, tanh

### Deep Learning Training (NEW in v2.0)
- **Autograd Engine**: Reverse-mode automatic differentiation with topological sort
- **Broadcasting**: Full NumPy-compatible broadcasting with `_unbroadcast` backward support
- **Loss Functions**: MSE loss, Cross-entropy loss, Softmax, Log-softmax
- **Neural Network Modules**: `nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Sequential`
- **Optimizers**: SGD (with momentum), Adam
- **Data Loading**: `DataLoader` with batching and shuffling

### GPU Acceleration
- 20 WGSL compute shaders for WebGPU
- N-Pass tree reduction (sum/max) with single CommandEncoder optimization
- Axis reduction for gradient computation
- Zero-copy reshape
- In-place parameter update kernel (axpy)
- TDR-safe workload chunking

---

## Installation

### From Source (any environment)

```bash
# Clone the repository
git clone https://github.com/ameva/tensor.git
cd tensor

# Install Python package
cd packages/forge-py
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Requirements
- **Python**: >= 3.9
- **NumPy**: >= 1.20.0
- **Node.js**: >= 18 (for WebGPU GPU acceleration)
- **Browser**: Chrome 113+ or Edge 113+ (for WebGPU in browser)

---

## Quick Start

### CPU Training (Pure Python + NumPy)

```python
import sys
sys.path.insert(0, 'packages/forge-py/src')

import numpy as np
import forge as at
from forge import nn, optim
from forge.functional import mse_loss

# XOR dataset
X = at.tensor(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32))
Y = at.tensor(np.array([[0],[1],[1],[0]], dtype=np.float32))

# Model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    pred = model(X)
    loss = mse_loss(pred, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f'Epoch {epoch}: loss = {loss.numpy():.6f}')
```

### Browser/Pyodide (GPU Acceleration)

```python
import forge as at

# Initialize WebGPU
await at.init()

# Create GPU tensors
x = at.tensor([[1, 2], [3, 4]], device='gpu')
w = at.random((2, 2), device='gpu', requires_grad=True)

# Forward pass on GPU
y = at.matmul(x, w)
y_cpu = await y.numpy_async()
print(y_cpu)
```

---

## Architecture

```
+--------------------------------------------------+
|  Python API (forge)                       |
|  nn.Module | optim | functional | autograd       |
+--------------------------------------------------+
|  Tensor Operations (ops.py)                      |
|  Broadcasting | Unbroadcast | Reduction          |
+--------------------------------------------------+
|  Lazy Graph Builder (graph.py)                   |
+--------------------------------------------------+
|  FFI Bridge (bridge.py <-> pyodideBridge.ts)     |
+--------------------------------------------------+
|  WebGPU Graph Executor (graphExecutor.ts)        |
|  N-Pass Reduction | Axis Reduction | Zero-copy   |
+--------------------------------------------------+
|  GPU Core (gpuCore.ts)                           |
|  20 WGSL Compute Shaders | Pipeline Cache       |
+--------------------------------------------------+
|  WebGPU Runtime (Chrome/Edge/Dawn)               |
+--------------------------------------------------+
```

---

## API Reference

### Tensor Creation
```python
at.tensor(data, device='cpu', requires_grad=False)
at.zeros(shape, device='cpu')
at.ones(shape, device='cpu')
at.random(shape, device='cpu')
at.full(shape, fill_value, device='cpu')
```

### Operations
```python
at.add(a, b)       # Broadcasting supported
at.sub(a, b)
at.mul(a, b)
at.div(a, b)
at.neg(a)
at.matmul(a, b)
at.transpose(a)
at.relu(a)
at.sigmoid(a)
at.tanh(a)
at.exp(a)
at.log(a)
at.sum(a)
at.mean(a)
at.reshape(a, new_shape)
```

### Neural Networks
```python
from forge import nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

params = model.parameters()
output = model(input_tensor)
```

### Optimizers
```python
from forge import optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Loss Functions
```python
from forge.functional import mse_loss, cross_entropy, softmax

loss = mse_loss(predictions, targets)
loss = cross_entropy(logits, labels)
probs = softmax(logits, axis=-1)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0.0 | 2026-08-12 | Deep learning training: autograd, broadcasting, reduction, loss functions, nn.Module, optimizers, DataLoader |
| v1.0.0 | 2026-08-11 | WebGPU inference: forward pass, 9 WGSL kernels, lazy graph executor |
| v0.1.0 | 2026-08-10 | Initial release: basic tensor operations |

---

## License

MIT License
