# Release Scope Matrix (Red Team Review 2)

## Release 1 (Public Technical Preview) Goal
"브라우저에서 Python API로 작은 2-layer MLP를 생성하고, 제한된 기본 연산과 autograd를 사용해 WebGPU에서 forward, backward, optimizer step을 정확하게 수행하는 Public Technical Preview"

### Core API (Must Include)
- **Tensor Creation**: `tensor`, `randn`, `zeros_like`
- **Math**: `add`, `sub`, `mul`, `div`, `neg`
- **Matrix**: 2D `matmul`
- **Activation**: `relu`
- **Reduction**: scalar `sum` (Needs fix)
- **Shape Ops**: `reshape`, `transpose` (Needs fix)
- **Training**: Basic Autograd, `SGD` optimizer
- **Memory**: `upload`, `readback`, `dispose` (Needs Quota fix)

### Experimental API
- **Advanced Math**: `exp`, `log`, `sigmoid`, `tanh`
- **Shape Ops**: `unsqueeze`, `squeeze`, `flatten`, generic `permute`
- **Reductions**: `max`, `mean`, `sum_axis`
- **NN Ops**: `dropout`
- **Broadcasting**

### Excluded from Release 1
- **Vision**: `Conv2d`, `maxpool2d`, `avgpool2d`, `im2col`, `col2im` (Require rigorous parity, stride/pad tests, and overlap gradients).
- **Advanced NN**: `BatchNorm2d`, `LayerNorm`, `Embedding`, `BMM`, `RNN`, `LSTM`, `MultiHeadAttention`, `TransformerEncoderLayer`.
- **Schedulers**: Composite LR schedulers.
- **Serialization**: Unverified roundtrip serialization.
