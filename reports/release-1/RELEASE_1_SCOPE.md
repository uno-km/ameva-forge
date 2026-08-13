# AMEVA-Forge Release 1 Scope

## Release 1 공식 목표
“브라우저에서 Pyodide Python API를 사용하여 작은 2-layer MLP를 생성하고, 제한된 핵심 Tensor 연산과 Autograd를 통해 WebGPU에서 forward, backward, optimizer step을 정확하고 반복 가능하게 실행하는 Public Technical Preview”

## 포함 대상 (Core)
- Tensor creation (zeros, ones, randn, full)
- Add, Sub, Mul, Div, Neg
- Exp, Log, ReLU, Sigmoid, Tanh
- Matmul, BMM
- Transpose, Reshape, Permute
- Sum, Mean, Max, Sum_axis
- Broadcasting
- Slice, Gather, Scatter
- Autograd Core (Topological sort, Gradient accumulation)
- SGD Optimizer

## 제외 사항 (Excluded)
- Conv2d, Pooling, BatchNorm2d (Requires more testing on GPU)
- RNN, LSTM, Transformer (Post-R1 scope due to Matmul bug and memory constraints)
- Native multithreading (Browser limit)

## Needs Verification
- Device lost recovery
- Memory leak after 1000 iterations
