from typing import List
from .tensor import Tensor
from .ops import random, zeros, transpose, relu, matmul
import numpy as np

class Module:
    def __init__(self):
        # VUL-011 Fix: object.__setattr__로 안전하게 초기화
        # __setattr__ 오버라이드 전에 내부 딕셔너리를 먼저 생성
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_params', {})
        object.__setattr__(self, 'training', True)
        
    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
            
    def eval(self):
        self.train(False)
    
    def forward(self, *args):
        raise NotImplementedError
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def parameters(self) -> List[Tensor]:
        params = list(self._params.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def state_dict(self, prefix='', keep_vars=False):
        from collections import OrderedDict
        state = OrderedDict()
        for name, param in self._params.items():
            key = prefix + name
            if keep_vars:
                state[key] = param
            else:
                state[key] = param.numpy() if hasattr(param, 'numpy') else param._data
                
        for name, module in self._modules.items():
            if module is not None:
                state.update(module.state_dict(prefix + name + '.', keep_vars))
                
        return state

    def load_state_dict(self, state_dict):
        my_state = self.state_dict(keep_vars=True)
        for name, param in my_state.items():
            if name in state_dict:
                val = state_dict[name]
                if hasattr(val, 'numpy'):
                    val = val.numpy()
                param._data = np.array(val, dtype=param._data.dtype if param._data is not None else np.float32)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Module):
            self._modules[name] = value
        if isinstance(value, Tensor) and getattr(value, 'requires_grad', False):
            self._params[name] = value
        object.__setattr__(self, name, value)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Kaiming initialization
        scale = (2.0 / in_features) ** 0.5
        w_data = np.random.randn(out_features, in_features).astype(np.float32) * scale
        self.weight = Tensor(shape=(out_features, in_features), dtype='float32', device='cpu', data=w_data, requires_grad=True)
        
        if bias:
            self.bias = Tensor(shape=(out_features,), dtype='float32', device='cpu',
                             data=np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x):
        out = matmul(x, transpose(self.weight))
        if self.bias is not None:
            from .ops import add
            out = add(out, self.bias)  # broadcasting: (batch, out) + (out,)
        return out


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Sigmoid(Module):
    def forward(self, x):
        from .ops import sigmoid
        return sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        from .ops import tanh_op
        return tanh_op(x)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        from .ops import max_pool2d
        return max_pool2d(x, self.kernel_size, self.stride, self.padding)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        from .ops import avg_pool2d
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding)

class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    def forward(self, x):
        from .ops import flatten
        return flatten(x, self.start_dim, self.end_dim)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.bias = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.zeros(num_features, dtype=np.float32), requires_grad=True)
        self.running_mean = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.zeros(num_features, dtype=np.float32), requires_grad=False)
        self.running_var = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.ones(num_features, dtype=np.float32), requires_grad=False)

    def forward(self, x):
        from .functional import batch_norm2d
        return batch_norm2d(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        from .ops import dropout
        return dropout(x, self.p, self.training)

class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        import math
        k = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        from .ops import random, tensor
        import numpy as np
        
        weight_data = np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.weight = tensor(weight_data, requires_grad=True)
        
        if bias:
            bias_data = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: 'Tensor') -> 'Tensor':
        from .ops import conv2d
        
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            self.weight.requires_grad = True
        if self.bias is not None and self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
            self.bias.requires_grad = True
            
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            from .ops import tensor
            import numpy as np
            self.weight = tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
            self.bias = tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        from .functional import layer_norm
        
        if self.weight is not None and self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            self.weight.requires_grad = True
        if self.bias is not None and self.bias.device != x.device:
            self.bias = self.bias.to(x.device)
            self.bias.requires_grad = True
            
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        from .functional import scaled_dot_product_attention
        from .ops import reshape, permute
        
        B, L, E = query.shape
        _, S, _ = key.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = permute(reshape(q, (B, L, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = permute(reshape(k, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = permute(reshape(v, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, is_causal, self.training)
        
        attn_out = reshape(permute(attn_out, (0, 2, 1, 3)), (B, L, E))
        
        return self.out_proj(attn_out)

class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        self.activation = ReLU()
        
    def forward(self, src, src_mask=None, is_causal=False):
        from .ops import add
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, is_causal=is_causal)
        src = add(src, self.dropout1(src2))
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = add(src, self.dropout2(src2))
        src = self.norm2(src)
        
        return src

class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import numpy as np
        from .ops import tensor
        
        pe = np.zeros((1, max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        
        self.pe = tensor(pe, requires_grad=False)
        
    def forward(self, x):
        from .ops import add, tensor
        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)
            
        seq_len = x.shape[1]
        pe_slice = tensor(self.pe._data[:, :seq_len, :], device=x.device)
        return add(x, pe_slice)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Standard normal initialization
        data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        from .ops import tensor
        self.weight = tensor(data, requires_grad=True)

    def forward(self, x):
        from .ops import embedding
        return embedding(self.weight, x)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        k = (1.0 / hidden_size) ** 0.5
        w_ih = np.random.uniform(-k, k, (hidden_size, input_size)).astype(np.float32)
        w_hh = np.random.uniform(-k, k, (hidden_size, hidden_size)).astype(np.float32)
        b_ih = np.random.uniform(-k, k, (hidden_size,)).astype(np.float32)
        b_hh = np.random.uniform(-k, k, (hidden_size,)).astype(np.float32)
        
        from .ops import tensor
        self.weight_ih = tensor(w_ih, requires_grad=True)
        self.weight_hh = tensor(w_hh, requires_grad=True)
        self.bias_ih = tensor(b_ih, requires_grad=True)
        self.bias_hh = tensor(b_hh, requires_grad=True)

    def forward(self, x, hx=None):
        from .ops import zeros, matmul, transpose, add, tanh_op
        
        if hx is None:
            hx = zeros((x.shape[0], self.hidden_size), device=x.device)
            
        # h_next = tanh(x @ weight_ih.T + bias_ih + hx @ weight_hh.T + bias_hh)
        term1 = add(matmul(x, transpose(self.weight_ih)), self.bias_ih)
        term2 = add(matmul(hx, transpose(self.weight_hh)), self.bias_hh)
        h_next = tanh_op(add(term1, term2))
        return h_next

class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        k = (1.0 / hidden_size) ** 0.5
        w_ih = np.random.uniform(-k, k, (4 * hidden_size, input_size)).astype(np.float32)
        w_hh = np.random.uniform(-k, k, (4 * hidden_size, hidden_size)).astype(np.float32)
        b_ih = np.random.uniform(-k, k, (4 * hidden_size,)).astype(np.float32)
        b_hh = np.random.uniform(-k, k, (4 * hidden_size,)).astype(np.float32)
        
        from .ops import tensor
        self.weight_ih = tensor(w_ih, requires_grad=True)
        self.weight_hh = tensor(w_hh, requires_grad=True)
        self.bias_ih = tensor(b_ih, requires_grad=True)
        self.bias_hh = tensor(b_hh, requires_grad=True)

    def forward(self, x, hx=None):
        from .ops import zeros, matmul, transpose, add, sigmoid, tanh_op, mul
        
        if hx is None:
            h = zeros((x.shape[0], self.hidden_size), device=x.device)
            c = zeros((x.shape[0], self.hidden_size), device=x.device)
        else:
            h, c = hx
            
        gates = add(
            add(matmul(x, transpose(self.weight_ih)), self.bias_ih),
            add(matmul(h, transpose(self.weight_hh)), self.bias_hh)
        )
        
        i_gate = sigmoid(gates[:, 0:self.hidden_size])
        f_gate = sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        g_gate = tanh_op(gates[:, 2*self.hidden_size:3*self.hidden_size])
        o_gate = sigmoid(gates[:, 3*self.hidden_size:4*self.hidden_size])
        
        c_next = add(mul(f_gate, c), mul(i_gate, g_gate))
        h_next = mul(o_gate, tanh_op(c_next))
        
        return h_next, c_next

class RNN(Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = RNNCell(input_size, hidden_size)
        
    def forward(self, x, hx=None):
        from .ops import cat, unsqueeze, permute
        if self.batch_first:
            x = permute(x, (1, 0, 2))
            
        seq_len = x.shape[0]
        outputs = []
        h = hx
        
        for t in range(seq_len):
            x_t = x[t]
            h = self.cell(x_t, h)
            outputs.append(unsqueeze(h, 0))
            
        out = cat(outputs, dim=0)
        
        if self.batch_first:
            out = permute(out, (1, 0, 2))
            
        return out, h

class LSTM(Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = LSTMCell(input_size, hidden_size)
        
    def forward(self, x, hx=None):
        from .ops import cat, unsqueeze, permute
        if self.batch_first:
            x = permute(x, (1, 0, 2))
            
        seq_len = x.shape[0]
        outputs = []
        
        if hx is None:
            h, c = None, None
        else:
            h, c = hx
            
        for t in range(seq_len):
            x_t = x[t]
            h, c = self.cell(x_t, (h, c) if h is not None else None)
            outputs.append(unsqueeze(h, 0))
            
        out = cat(outputs, dim=0)
        
        if self.batch_first:
            out = permute(out, (1, 0, 2))
            
        return out, (h, c)
