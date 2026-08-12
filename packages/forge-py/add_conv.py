import sys

with open('c:/Users/GAME/Desktop/uno-km/dev/AMEVA-Tensor/packages/forge-py/src/forge/ops.py', 'a', encoding='utf-8') as f:
    f.write('''

class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, weight: Tensor, bias: Optional[Tensor], stride: int, padding: int) -> Tensor:
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        
        N, C, H, W = x.shape
        C_out, C_in, K_h, K_w = weight.shape
        if C != C_in:
            raise AMEVAForgeShapeError(f"Input channels {C} does not match weight channels {C_in}")
            
        H_out = (H + 2 * padding - K_h) // stride + 1
        W_out = (W + 2 * padding - K_w) // stride + 1
        
        if _should_use_gpu(x, weight):
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", inputs=[x], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            weight_t = weight_reshaped.transpose(0, 1)
            
            out_2d = Tensor(shape=(N * H_out * W_out, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="matmul", inputs=[x_col, weight_t], params=[N * H_out * W_out, C_out, C * K_h * K_w])
                            
            out = out_2d.reshape((N, H_out * W_out, C_out)).transpose(1, 2).reshape((N, C_out, H_out, W_out))
            if bias is not None:
                bias_reshaped = bias.reshape((1, C_out, 1, 1))
                out = out + bias_reshaped
            return out
        else:
            x_data = _require_cpu_data(x)
            weight_data = _require_cpu_data(weight)
            
            x_col = np.zeros((N, H_out * W_out, C * K_h * K_w), dtype=np.float32)
            for n in range(N):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        patch = np.zeros((C, K_h, K_w), dtype=np.float32)
                        for c in range(C):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    h_in = h_start + k_h
                                    w_in = w_start + k_w
                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        patch[c, k_h, k_w] = x_data[n, c, h_in, w_in]
                        x_col[n, h_out * W_out + w_out, :] = patch.flatten()
            
            weight_reshaped = weight_data.reshape((C_out, C * K_h * K_w))
            out_data = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
            for n in range(N):
                out_2d = x_col[n] @ weight_reshaped.T
                out_data[n] = out_2d.T.reshape((C_out, H_out, W_out))
                
            if bias is not None:
                bias_data = _require_cpu_data(bias)
                out_data += bias_data.reshape((1, C_out, 1, 1))
                
            return Tensor(shape=(N, C_out, H_out, W_out), dtype=x.dtype, device="cpu", requires_grad=False, data=out_data)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        N, C, H, W = x.shape
        C_out, C_in, K_h, K_w = weight.shape
        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]
        
        grad_bias = None
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(axis=(0, 2, 3)).reshape(bias.shape)
            
        if _should_use_gpu(x, weight):
            grad_out_2d = grad_output.transpose(1, 2).transpose(2, 3).reshape((N * H_out * W_out, C_out))
            x_col = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                           op="im2col", inputs=[x], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            x_col_t = x_col.transpose(0, 1)
            grad_weight_2d = Tensor(shape=(C * K_h * K_w, C_out), dtype=x.dtype, device="gpu", requires_grad=False,
                                    op="matmul", inputs=[x_col_t, grad_out_2d], params=[C * K_h * K_w, C_out, N * H_out * W_out])
            grad_weight = grad_weight_2d.transpose(0, 1).reshape(weight.shape)
            
            weight_reshaped = weight.reshape((C_out, C * K_h * K_w))
            grad_x_col_2d = Tensor(shape=(N * H_out * W_out, C * K_h * K_w), dtype=x.dtype, device="gpu", requires_grad=False,
                                   op="matmul", inputs=[grad_out_2d, weight_reshaped], params=[N * H_out * W_out, C * K_h * K_w, C_out])
            
            grad_x = Tensor(shape=(N, C, H, W), dtype=x.dtype, device="gpu", requires_grad=False,
                            op="col2im", inputs=[grad_x_col_2d], params=[N, C, H, W, K_h, K_w, stride, padding, H_out, W_out])
            
            return grad_x, grad_weight, grad_bias, None, None
        else:
            x_data = _require_cpu_data(x)
            weight_data = _require_cpu_data(weight)
            grad_out_data = _require_cpu_data(grad_output)
            
            x_col = np.zeros((N, H_out * W_out, C * K_h * K_w), dtype=np.float32)
            for n in range(N):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        patch = np.zeros((C, K_h, K_w), dtype=np.float32)
                        for c in range(C):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    h_in = h_start + k_h
                                    w_in = w_start + k_w
                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        patch[c, k_h, k_w] = x_data[n, c, h_in, w_in]
                        x_col[n, h_out * W_out + w_out, :] = patch.flatten()
            
            grad_weight_data = np.zeros_like(weight_data)
            grad_x_data = np.zeros_like(x_data)
            weight_reshaped = weight_data.reshape((C_out, C * K_h * K_w))
            
            grad_out_2d = grad_out_data.transpose(0, 2, 3, 1).reshape(N, H_out * W_out, C_out)
            
            grad_x_col = np.zeros_like(x_col)
            for n in range(N):
                gw = x_col[n].T @ grad_out_2d[n]
                grad_weight_data += gw.T.reshape(weight.shape)
                
                gxc = grad_out_2d[n] @ weight_reshaped
                grad_x_col[n] = gxc
                
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        patch = grad_x_col[n, h_out * W_out + w_out].reshape(C, K_h, K_w)
                        h_start = h_out * stride - padding
                        w_start = w_out * stride - padding
                        for c in range(C):
                            for k_h in range(K_h):
                                for k_w in range(K_w):
                                    h_in = h_start + k_h
                                    w_in = w_start + k_w
                                    if 0 <= h_in < H and 0 <= w_in < W:
                                        grad_x_data[n, c, h_in, w_in] += patch[c, k_h, k_w]
                                        
            grad_x = Tensor(shape=x.shape, dtype=x.dtype, device="cpu", requires_grad=False, data=grad_x_data)
            grad_weight = Tensor(shape=weight.shape, dtype=weight.dtype, device="cpu", requires_grad=False, data=grad_weight_data)
            
            return grad_x, grad_weight, grad_bias, None, None

def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0) -> Tensor:
    return Conv2dFunction.apply(x, weight, bias, stride, padding)
''')

with open('c:/Users/GAME/Desktop/uno-km/dev/AMEVA-Tensor/packages/forge-py/src/forge/nn.py', 'a', encoding='utf-8') as f:
    f.write('''

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
        
        weight_data = np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size))
        self.weight = tensor(weight_data, requires_grad=True)
        
        if bias:
            bias_data = np.random.uniform(-k, k, (out_channels,))
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
''')
